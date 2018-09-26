
import keras
import logging
import  numpy as np
import tensorflow as tf

from collections import OrderedDict, deque
from keras.layers import Input, Dense, Lambda, Dropout, Dot, Permute, Reshape, Embedding, Concatenate, Multiply
from keras.models import Model, load_model


import os
import sys
sys.path.append('../utils/')
sys.path.append('../metrics/')

from file_operation import write_result_to_trec_format, load_pickle, retain_file
from evaluations import evaluate_trec
from model import BasicModel, NBatchLogger
from nprf_knrm_config import NPRFKNRMConfig
from nprf_knrm_pair_generator import NPRFKNRMPairGenerator
from relevance_info import Relevance
from result import Result

from rank_losses import rank_hinge_loss

class NPRFKNRM(BasicModel):

  def __init__(self, config):
    super(NPRFKNRM, self).__init__(config)
    self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=118)

  def build(self):
    # qd_input = Input((self.config.kernel_size,), name="qd_input")
    dd_input = Input((self.config.nb_supervised_doc, self.config.kernel_size), name='dd_input')
    # z = Dense(self.config.hidden_size, activation='tanh', name="qd_hidden")(qd_input)
    # qd_out = Dense(self.config.out_size, name="qd_out")(z)

    z = Dense(self.config.hidden_size, activation='tanh', name="dd_hidden")(dd_input)
    dd_init_out = Dense(self.config.out_size, name='dd_init_out')(z)

    dd_gate = Input((self.config.nb_supervised_doc, 1), name='baseline_doc_score')
    dd_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False, name='dd_gate')(dd_gate)
    # dd_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config.nb_supervised_doc,), name='dd_softmax')(dd_w)

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

    model = Model(inputs=[dd_input, dd_gate], outputs=[out])
    print(model.summary())

    return model


  def train_wrapper(self, fold, output_file,):
    pair_generator = NPRFKNRMPairGenerator(**self.config.generator_params)
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
    # generate re rank features
    dd_d, score_gate, len_indicator = \
        pair_generator.generate_list_batch(qualified_qid_list, self.config.rerank_topk)

    return [dd_d, score_gate], len_indicator, res_dict, qualified_qid_list

  def eval_by_qid_list(self, X, len_indicator, res_dict, qualified_qid_list,  model,
                       relevance_dict, rerank_topk, nb_supervised_doc, doc_topk_term, qrels_file,
                       docnolist_file, runid, output_file, ):
    # qd_d, dd_d, score_gate = X
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
  conf = NPRFKNRMConfig()
  ddmknrm = NPRFKNRM(conf)

  # ddm.build()
  # ddm.build2()
  argv = sys.argv
  phase = argv[1]
  if phase == '--fold':
    fold = int(argv[2])
    temp = argv[3]
  else:
    fold = 1
    temp = 'temp'

  ddmknrm.train_wrapper(fold, temp)
