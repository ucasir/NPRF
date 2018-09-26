
import time
import keras
import logging
import multiprocessing
import  numpy as np
import cPickle as pickle
import tensorflow as tf

from functools import partial
from contextlib import contextmanager
from multiprocessing import Process, Queue
from collections import OrderedDict, deque
from keras.layers import Input, Dense, Lambda, Dropout, Dot, Permute, Reshape, Embedding, Concatenate, Multiply
from keras.activations import softmax, tanh
from keras.models import Model, load_model
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.backend.tensorflow_backend import set_session

import os
import sys
sys.path.append('../utils/')
sys.path.append('../metrics/')

from file_operation import write_result_to_trec_format, load_pickle, retain_file
from evaluations import evaluate_trec
from model import BasicModel, NBatchLogger
from relevance_info import Relevance
from result import Result
from nprf_drmm_pair_generator import NPRFDRMMPairGenerator
from nprf_drmm_config import NPRFDRMMConfig
from rank_losses import rank_hinge_loss



class NPRFDRMM(BasicModel):

  def __init__(self, config):
    super(NPRFDRMM, self).__init__(config)
    self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=118)
    self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=118)

  def build(self):

    dd_q_input = Input((self.config.nb_supervised_doc, self.config.doc_topk_term, 1), name='dd_q_input')
    dd_d_input = Input((self.config.nb_supervised_doc, self.config.doc_topk_term,
                        self.config.hist_size), name='dd_d_input')

    dd_q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False, name='dd_q_gate')(dd_q_input)
    dd_q_w = Lambda(lambda x: softmax(x, axis=2), output_shape=(
                  self.config.nb_supervised_doc, self.config.doc_topk_term,), name='dd_q_softmax')(dd_q_w)

    z = dd_d_input
    for i in range(self.config.nb_layers):
      z = Dense(self.config.hidden_size[i], activation='tanh',
                kernel_initializer=self.initializer_fc, name='hidden')(z)
    z = Dense(self.config.out_size, kernel_initializer=self.initializer_fc, name='dd_d_gate')(z)
    z = Reshape((self.config.nb_supervised_doc, self.config.doc_topk_term,))(z)
    dd_q_w = Reshape((self.config.nb_supervised_doc, self.config.doc_topk_term,))(dd_q_w)
    # out = Dot(axes=[2, 2], name='dd_pseudo_out')([z, dd_q_w])

    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]), name='dd_pseudo_out')([z, dd_q_w])
    dd_init_out = Lambda(lambda x: tf.matrix_diag_part(x), output_shape=(self.config.nb_supervised_doc,), name='dd_init_out')(out)
    '''
    dd_init_out = Lambda(lambda x: tf.reduce_sum(x, axis=2), output_shape=(self.config.nb_supervised_doc,))(z)
    '''
    #dd_out = Reshape((self.config.nb_supervised_doc,))(dd_out)

    # dd out gating
    dd_gate = Input((self.config.nb_supervised_doc, 1), name='baseline_doc_score')
    dd_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False, name='dd_gate')(dd_gate)
    # dd_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config.nb_supervised_doc,), name='dd_softmax')(dd_w)

    # dd_out = Dot(axes=[1, 1], name='dd_out')([dd_init_out, dd_w])
    dd_w = Reshape((self.config.nb_supervised_doc,))(dd_w)
    dd_init_out = Reshape((self.config.nb_supervised_doc,))(dd_init_out)


    if self.config.method in [1, 3]: # no doc gating, with dense layer
      z = dd_init_out
    elif self.config.method == 2:
      logging.info("Apply doc gating")
      z = Multiply(name='dd_out')([dd_init_out, dd_w])
    else:
      raise ValueError("Method not initialized, please check config file")

    if self.config.method in [1, 2]:
      logging.info("Dense layer on top")
      z = Dense(self.config.merge_hidden, activation='tanh', name='merge_hidden')(z)
      out = Dense(self.config.merge_out, name='score')(z)
    else:
      logging.info("Apply doc gating, No dense layer on top, sum up scores")
      out = Dot(axes=[1, 1], name='score')([z, dd_w])

    model = Model(inputs=[dd_q_input, dd_d_input, dd_gate], outputs=[out])
    print(model.summary())

    return model

  def train_wrapper(self, fold, output_file,):
    pair_generator = NPRFDRMMPairGenerator(**self.config.generator_params)
    model = self.build()
    # adagrad
    model.compile(optimizer=self.config.optimizer, loss=rank_hinge_loss)

    eval_met = self.train(model, pair_generator, fold, output_file, use_nprf=True)

    return eval_met

  def eval_by_qid_list_helper(self, qid_list, pair_generator):

    relevance_dict = load_pickle(self.config.relevance_dict_path)
    qid_list = sorted(qid_list)
    qualified_qid_list = []
    res_dict = OrderedDict()
    for qid in qid_list:
      relevance = relevance_dict.get(qid)
      supervised_docid_list = relevance.get_supervised_docid_list()
      if len(supervised_docid_list) < self.config.nb_supervised_doc:
        # cannot construct d2d feature, thus not need to be update
        score_list = relevance.get_supervised_score_list()
        res = Result(qid, supervised_docid_list, score_list, self.config.runid)
        res_dict.update({qid: res})
        logging.warn("query {0} not to be rerank".format(qid))
      else:
        qualified_qid_list.append(qid)
    # generate re rank score
    dd_q, dd_d, score_gate, len_indicator = \
                          pair_generator.generate_list_batch(qualified_qid_list, self.config.rerank_topk)

    return [dd_q, dd_d, score_gate], len_indicator, res_dict, qualified_qid_list

  def eval_by_qid_list(self, X, len_indicator, res_dict, qualified_qid_list,  model,
                       relevance_dict, rerank_topk, nb_supervised_doc, doc_topk_term, qrels_file,
                       docnolist_file, runid, output_file, ):
    # dd_q, dd_d = list(map(lambda x: x[:, :nb_supervised_doc, : doc_topk_term, :], [dd_q, dd_d]))
    topk_score_all = model.predict_on_batch(X)
    topk_score_all = topk_score_all.flatten()

    for i, qid in enumerate(qualified_qid_list):
      relevance = relevance_dict.get(qid)
      supervised_docid_list = relevance.get_supervised_docid_list()
      topk_score = topk_score_all[sum(len_indicator[:i]): sum(len_indicator[:i]) + len_indicator[i]]

      if len(supervised_docid_list) <= rerank_topk:
        score_list = topk_score
      else:
        behind_score = np.min(topk_score) - 0.001 - np.sort(np.random.random((len(supervised_docid_list) - rerank_topk,)))
        score_list = np.concatenate((topk_score, behind_score))

      res = Result(qid, supervised_docid_list, score_list, runid)
      res.update_ranking()
      res_dict.update({qid: res})
    # print "generate score {0}".format(time.time()-t)
    write_result_to_trec_format(res_dict, output_file, docnolist_file)
    met = evaluate_trec(qrels_file, output_file)

    return met


if __name__ == '__main__':
  argv = sys.argv
  phase = argv[1]
  if phase == '--fold':
    fold = int(argv[2])
    temp = argv[3]
  else:
    fold = 1
    temp = 'temp'

  config = NPRFDRMMConfig()
  ddm = NPRFDRMM(config)
  ddm.train_wrapper(fold, temp)

