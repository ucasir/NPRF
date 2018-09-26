#!/usr/bin/python  
# -*- coding: utf-8 -*-  

import os
import sys
import logging

import numpy as np
from collections import OrderedDict, Counter
from gensim.models.keyedvectors import KeyedVectors

import multiprocessing
from functools import partial
from contextlib import contextmanager

sys.path.append("../utils")
from matrix import similarity_matrix, hist_from_matrix, kernel_from_matrix, kernal_mus, kernel_sigmas
from file_operation import parse_corpus, parse_content, parse_topk_content, load_pickle, save_pickle, \
  make_directory, parse_topic, parse_stoplist, df_map_from_file
from relevance_info import Relevance

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def sim_mat_and_kernel_d2d(relevance_file, topic_file, corpus_file, topk_corpus_file, embedding_file, stop_file,
                           sim_output_path, kernel_output_path, kernel_mu_list, kernel_sigma_list,
                           topk_supervised, d2d, test):
  '''Simultaneously compute similarity matrix and RBF kernel features

  Args:
    relevance_file: A dumped relevance dict file
    topic_file: a single line format topic file. format: qid term1 term2 ...
    corpus_file: corpus corresponding to docnolist file. format: docno\tdoclen\tterm1 term2
    topk_corpus_file: corpus that contain only the topk terms for each document, format: same as corpus_file
    embedding_file: output file from word2vec toolkit, boolean=True
    stop_file: a stopword list file, one word per line
    sim_output_path:
    kernel_output_path:
    kernel_mu_list:
    kernel_sigma_list:
    topk_supervised: number of top-n documents for each query
    d2d: True for NPRF, False for simple query-document matching used by e.g. DRMM, K-NRM
    test: control the temporary output. Set false

  Returns:

  '''
  relevance_dict = load_pickle(relevance_file)
  topic_dict = parse_topic(topic_file)
  corpus = parse_corpus(corpus_file)
  topk_corpus = parse_corpus(topk_corpus_file)

  embeddings = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
  stoplist = parse_stoplist(stop_file)
  qid_list = relevance_dict.keys()



  for qid in qid_list:
    sim_mat_and_kernel_per_query(relevance_dict, topic_dict, corpus, topk_corpus, embeddings, stoplist, sim_output_path,
                                 kernel_output_path, kernel_mu_list, kernel_sigma_list, topk_supervised, d2d, test, qid)


def sim_mat_and_kernel_per_query(relevance_dict, topic_dict,  corpus, topk_corpus, embeddings, stoplist, sim_output_path,
                                 kernel_output_path, kernel_mu_list,
                                 kernel_sigma_list, topk_supervised, d2d, test, qid):
  relevance = relevance_dict.get(qid)
  topic_content = topic_dict.get(qid)
  supervised_docid_list = relevance.get_supervised_docid_list()
  topk_supervised_docid_list = supervised_docid_list[: topk_supervised]
  if len(topk_supervised_docid_list) < topk_supervised:
    logging.warn("{0} does not have enough supervised documents, in total {1}.".format(
      qid, len(topk_supervised_docid_list)))

  # hist_output_dir = make_directory(hist_output_path, str(qid))
  sim_output_dir = make_directory(sim_output_path, str(qid))
  ker_output_dir = make_directory(kernel_output_path, str(qid))

  OOV_dict = OrderedDict()
  judged_docid_list = relevance.get_judged_docid_list()
  '''
    because we only want to rerank top 500 docs, but judged docs that lie in
    in top 1000 should also be considered, for the sufficiency of training 
  '''
  cand = judged_docid_list[0] + judged_docid_list[1] + judged_docid_list[2]
  waitlist = [docid for docid in cand if docid in supervised_docid_list[500:2000]]
  useful_docid_list = supervised_docid_list[: 1000]# [:500] + waitlist
  for docid in useful_docid_list:
    sim_mat_list = []
    ker_list = []
    doc_content = parse_topk_content(corpus[docid]) # for pacrr
    # doc_content = parse_content(corpus[docid], stoplist)
    # doc_content = parse_content(corpus[docid])

    if d2d:
      sim_file_name = os.path.join(sim_output_dir, 'q{0}_d{1}.pickle'.format(qid, docid))
    else:
      sim_file_name =  os.path.join(sim_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))
    ker_file_name = os.path.join(ker_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))

    if os.path.exists(ker_file_name):
      pass
    else:
      if d2d:
        for sup_docid in topk_supervised_docid_list:
          sup_doc_content = parse_topk_content(topk_corpus[sup_docid])[:30]
          sim_mat = similarity_matrix(sup_doc_content, doc_content, embeddings, OOV_dict)[:, :20000]
          kernel_feat = kernel_from_matrix(sim_mat, kernel_mu_list, kernel_sigma_list, d2d)
          sim_mat_list.append(sim_mat.astype(np.float16))
          ker_list.append(kernel_feat)
          if test == True:
            print(qid, docid, sup_docid)
            print(doc_content)
            print(sup_doc_content)
            assert 1 == 2

        ker_list = np.asarray(ker_list)
        save_pickle(sim_mat_list, sim_file_name)
        np.save(ker_file_name, ker_list)
      else:
        if test == True:
          print(qid, docid)
          print(topic_content)
          print(doc_content)
          assert 1 == 2
        sim_mat = similarity_matrix(topic_content, doc_content, embeddings, OOV_dict)
        kernel_feat = kernel_from_matrix(sim_mat, kernel_mu_list, kernel_sigma_list, d2d)

        np.save(sim_file_name, sim_mat)
        np.save(ker_file_name, kernel_feat)

  logging.info("Finish for topic {0}".format(qid))


def hist_d2d(relevance_file, text_max_len, hist_size, sim_path, hist_path, d2d=False):
  # qid_list = os.listdir(sim_path)
  relevance_dict = load_pickle(relevance_file)
  qid_list = relevance_dict.keys()
  with poolcontext(processes=14) as pool:
    pool.map(partial(hist_per_query, relevance_dict, text_max_len, hist_size, sim_path, hist_path, d2d), qid_list)
  logging.info("Finish all!")


def hist_per_query(relevance_dict, text_max_len, hist_size, sim_path, hist_path, d2d, qid):
  # relevance_dict = load_pickle(relevance_file)
  hist_output_dir = make_directory(hist_path, str(qid))
  # files = glob.glob(os.path.join(sim_path, str(qid), '*.pickle'))
  relevance = relevance_dict.get(qid)
  supervised_docid_list = relevance.get_supervised_docid_list()
  judged_docid_list = relevance.get_judged_docid_list()
  '''
    because we only want to rerank top 500 docs, but judged docs that lie in
    in top 1000 should also be considered, for the sufficiency of training 
  '''
  cand = judged_docid_list[0] + judged_docid_list[1] + judged_docid_list[2]
  waitlist = [docid for docid in cand if docid in supervised_docid_list[500:2000]]
  useful_docid_list = supervised_docid_list[: 1000]#[:500] + waitlist

  for docid in useful_docid_list: # supervised_docid_list[:1000]: # supervised_docid_list[:500] + waitlist:
    # file_name = os.path.basename(sim_file)
    # file_name = re.sub('pickle', 'npy', file_name)
    # _file = os.path.join(hist_output_dir, file_name)
    if d2d:
      sim_file_name = os.path.join(sim_path, str(qid), 'q{0}_d{1}.pickle'.format(qid, docid))
    else:
      sim_file_name = os.path.join(sim_path, str(qid), 'q{0}_d{1}.npy'.format(qid, docid))
    hist_file_name = os.path.join(hist_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))

    if os.path.exists(hist_file_name):
      pass
    else:
      if d2d:
        sim_list = load_pickle(sim_file_name)
        hist_array = np.zeros((len(sim_list), text_max_len, hist_size), dtype=np.float32)
        for i, sim_mat in enumerate(sim_list):
          sim_mat = sim_mat[:, :20000]
          hist = hist_from_matrix(text_max_len, hist_size, sim_mat)
          hist_array[i] = hist

        np.save(hist_file_name, hist_array)
      else:
        sim_mat = np.load(sim_file_name)
        hist = hist_from_matrix(text_max_len, hist_size, sim_mat)
        np.save(hist_file_name, hist)
  logging.info("Finish for topic {0}".format(qid))


@contextmanager
def poolcontext(*args, **kwargs):
  pool = multiprocessing.Pool(*args, **kwargs)
  yield pool
  pool.terminate()


def topk_term(df_file, corpus_file, output_file, nb_docs, topk):
  '''Get top-k tf-idf weighting terms for each document

  Args:
    df_file: format: docno\tdf\cf
    corpus_file:
    output_file: format: docno\tdoclen\tterm
    nb_docs: number of total documents in the collection
    topk: number of top terms

  Returns:

  '''

  def representsInt(s):
    try:
      int(s)
      return True
    except ValueError:
      return False

  df_map = df_map_from_file(df_file)
  # stop_list = parse_stoplist(stop_file)

  def formatted(docno, doclen, term_list):
    term_string = ' '.join(term_list)
    line = "{0}\t{1}\t{2}".format(docno, doclen, term_string)

    return line

  info_list = []
  i = 0
  with open(corpus_file, 'r') as f:
    for line in f:
      if i % 1000 == 0:
        logging.info("doc {0}".format(i))
      i += 1
      term_list = []
      score_list = []
      docno, doclen, raw_content = line.strip().split('\t')
      term_count_pair = Counter(raw_content.split()).most_common()
      # term_count_pair = list(map(lambda x: x.split(':'), raw_content.split()))

      for term, count in term_count_pair:
        count = int(count)
        if df_map.get(term) == None:
        # if term in stop_list or df_map.get(term) == None:
          pass
        else:
          df = df_map.get(term)
          idf = np.log((nb_docs - df + 0.5) / (df + 0.5))
          tfidf = count * idf
          score_list.append(tfidf)
          term_list.append(term)
      # sort the list
      tuple_list = zip(term_list, score_list)
      sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)
      topk_terms = zip(*sorted_tuple_list)[0]
      # add to
      qualified_terms = []
      index = 0
      while len(qualified_terms) < topk and index < len(topk_terms):
        t = topk_terms[index]
        if not representsInt(t) and len(t) >= 2:
          qualified_terms.append(t)
        index += 1
      if len(qualified_terms) < topk:
        logging.warn('document {0} does not contain enough terms, totally {1}'.format(docno, len(qualified_terms)))
      posting = formatted(docno, doclen, qualified_terms)
      info_list.append(posting)

  res = '\n'.join(info_list)
  with open(output_file, 'w') as f:
    f.write(res)



def parse_idf_for_document(relevance_file, df_file, document_file, output_file,
                           rerank_topk=500, doc_topk_term=30, nb_doc=528155):
  '''Get the idf weight for top k terms in each document

  Args:
    relevance_file:
    df_file:
    document_file:
    output_file:
    rerank_topk:
    doc_topk_term:
    nb_doc:

  Returns:

  '''
  relevance_dict = load_pickle(relevance_file)
  df_map = df_map_from_file(df_file)
  topk_term_corpus = parse_corpus(document_file)

  idf_map = OrderedDict()
  for qid, relevance in relevance_dict.items():
    #relevance = relevance_dict.get(qid)
    logging.info("query {0}".format(qid))
    supervised_docid_list = relevance.get_supervised_docid_list()[: rerank_topk]
    curr_idf = parse_idf_per_query(supervised_docid_list, df_map, doc_topk_term, topk_term_corpus, nb_doc)
    idf_map.update({qid: curr_idf})

  save_pickle(idf_map, output_file)


def parse_idf_per_query(supervised_docid_list, df_map, doc_topk_term, topk_corpus, nb_doc):

  idf_map = OrderedDict()
  for docid in supervised_docid_list:
    doc_topk = parse_topk_content(topk_corpus[docid])
    df = np.asarray([df_map.get(t) if df_map.get(t) != None else 1 for t in doc_topk])
    idf = np.log((nb_doc - df + 0.5) / (df + 0.5))
    idf_pad = np.zeros((doc_topk_term, ), dtype=np.float32)
    idf_pad[:len(idf)] = idf
    idf_map.update({docid: idf_pad})

  return idf_map

