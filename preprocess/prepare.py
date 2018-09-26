
import os
import logging
import warnings
import numpy as np
import cPickle as pickle

import sys
sys.path.append("../utils")

from file_operation import load_pickle, parse_content
from collections import OrderedDict
from bs4 import BeautifulSoup
from relevance_info import Relevance
from file_operation import docno2docid, df_map_from_file


def update_qrels_relevance(qrels_file, docno_map):
  '''Update docnos in qrels file

  Args:
    qrels_file:
    docno_map:

  Returns:

  '''
  with open(qrels_file, 'r') as f:
    qrels = f.readlines()

  qrels_relevance_dict = OrderedDict()
  prev_qid = -1
  relevance_posting = [[], [], []]
  for line in qrels:
    tokens = line.strip().split()
    qid, docno, relevance = tokens[0], tokens[2], tokens[3]
    docid = docno_map.get(docno)

    if docid == None:
      pass
    else:
      relevance, qid = int(relevance), int(qid)
      if relevance > 2 or relevance < 0:
        continue
      if qid != prev_qid:
        if sum(map(len, relevance_posting)) > 0:
          qrels_relevance_dict.update({prev_qid: relevance_posting})
        prev_qid = qid
        relevance_posting = [[], [], []]
        relevance_posting[relevance].append(docid)
      else:
        relevance_posting[relevance].append(docid)
  qrels_relevance_dict.update({prev_qid: relevance_posting})

  return qrels_relevance_dict

def update_qrels_from_res_and_qrels(qrels_file, docno_map, res_dict):
  '''Only select docnos that occur in the result file

  Args:
    qrels_file:
    docno_map:
    res_dict:

  Returns:

  '''
  with open(qrels_file, 'rb') as f:
    qrels = f.readlines()


  qrels_relevance_dict = OrderedDict() # only from qrels
  prev_qid = -1

  relevance_map = OrderedDict()
  # update from qrels
  for line in qrels:
    tokens = line.strip().split()
    qid, docno, relevance_score = tokens[0], tokens[2], tokens[3]
    docid = docno_map.get(docno)
    if docid == None: # ignore those that are not in the result list
      pass
    else:
      relevance_score, qid = int(relevance_score), int(qid)
      if qid != prev_qid:
        if len(relevance_map.values()) > 0:
          qrels_relevance_dict.update({prev_qid: relevance_map})
        prev_qid = qid
        relevance_map = OrderedDict()
        relevance_map.update({docid: relevance_score})
      else:
        relevance_map.update({docid: relevance_score})
  qrels_relevance_dict.update({prev_qid: relevance_map})

  # update from result
  relevance_all = OrderedDict() # from top 1000 result
  for qid in qrels_relevance_dict.keys():
    curr_supervised_docid_list = res_dict.get(qid)[0] # (docid_list, score_list)
    curr_qrels_map = qrels_relevance_dict.get(qid)
    relevance_posting = [[], [], []]
    for docid in curr_supervised_docid_list[:1000]:
      relevance_score = curr_qrels_map.get(docid)
      if relevance_score == None:
        relevance_score = 0
      elif relevance_score > 2:
        relevance_score = 2
      elif relevance_score < 0:
        relevance_score = 0
      else:
        pass
      relevance_posting[relevance_score].append(docid)
    if len(relevance_posting[1]) + len(relevance_posting[2]) < 5:
      logging.warn("topic {0}: relevant document less than 5".format(qid))
    relevance_all.update({qid: relevance_posting})

  return relevance_all


def update_res_relevance(res_file, docno_map):

  with open(res_file, 'r') as f:
    res = f.readlines()

  #res_topk = []
  #for i in range(len(res)/result_per_query):
  #  res_topk.extend(res[i*result_per_query: i*result_per_query + k])
  logging.info("There are {0} lines to be read.".format(len(res)))

  res_relevance_dict = OrderedDict()
  prev_qid = -1
  docid_list, score_list = [], []

  for line in res:
    tokens = line.strip().split()
    qid, docno, rank, score = int(tokens[0]), tokens[2], int(tokens[3]), float(tokens[4])
    docid = docno_map.get(docno)
    if docid == None:
      logging.warn("Cannot get docid for docno {0}".format(docno))
    #if rank < k:
    #  docid = docno_map.get(docid)
    if qid != prev_qid:
      if len(docid_list) > 0:
        res_relevance_dict.update({prev_qid: (docid_list, score_list)})
      prev_qid = qid
      docid_list, score_list = [], []
      docid_list.append(docid)
      score_list.append(score)
    else:
      docid_list.append(docid)
      score_list.append(score)
  res_relevance_dict.update({prev_qid: (docid_list, score_list)})

  return res_relevance_dict


def create_relevance(res_file, qrels_file, docnolist_file, output_file):
  '''

  Args:
    res_file: a standard TREC result file
    qrels_file: a standard TREC qrels file
    docnolist_file: unique docnos in a file, one docno per line
    output_file:

  Returns:
    A dumped relevance dict file
  '''
  docno_map = docno2docid(docnolist_file, 0)
  relevance_dict = OrderedDict()

  # qrels_relevance_dict = update_qrels_relevance(qrels_file, docno_map)

  res_relevance_dict = update_res_relevance(res_file, docno_map)

  qrels_relevance_dict = update_qrels_from_res_and_qrels(qrels_file, docno_map, res_relevance_dict)
  # qrels_relevance_dict = update_qrels_relevance(qrels_file, docno_map)

  for qid in qrels_relevance_dict.keys():
    supervised_docid_list, supervised_score_list = res_relevance_dict.get(qid)
    judged_docid_list = qrels_relevance_dict.get(qid)
    judged_docid_list_within_supervised = []
    for docid_list in judged_docid_list:
      cand = [docid for docid in docid_list if docid in supervised_docid_list[:1000]]
      judged_docid_list_within_supervised.append(cand)
    '''
    for i in range(len(judged_docid_list)):
      for docid in judged_docid_list[i]:
        if docid not in supervised_docid_list:
          judged_docid_list[i].remove(docid)
    '''
    relevance = Relevance(qid, judged_docid_list_within_supervised, supervised_docid_list, supervised_score_list)
    relevance_dict.update({qid: relevance})

  with open(output_file, 'w') as f:
    pickle.dump(relevance_dict, f)


def parse_idf_for_query(df_file, topic_file, output_file, maxlen, nb_doc=25205179):
  '''extract idf of each term in the query

  Args:
    df_file:
    topic_file:
    output_file:
    maxlen:
    nb_doc:

  Returns:

  '''
  df_map = df_map_from_file(df_file)
  print("df map initialized done.")

  with open(topic_file, 'rb') as f:
    topics = f.readlines()

  idf_map = OrderedDict()
  for line in topics:
    tokens = line.strip().split('\t')
    qid = int(tokens[0])
    df = np.asarray([df_map.get(t) if df_map.get(t) != None else 1 for t in tokens[1:]])
    idf = np.log((nb_doc - df + 0.5)/(df + 0.5))
    idf_pad = np.zeros((maxlen, ), dtype=np.float32)
    idf_pad[:len(idf)] = idf
    idf_map.update({qid: idf_pad})

  with open(output_file, 'wb') as f:
    pickle.dump(idf_map, f)

def get_query_length(topic_file):
  l = 0
  with open(topic_file, 'rb') as f:
    for line in f:
      tokens = line.strip().split()
      print(tokens)
      l = max(l, len(tokens) - 1)

  print("Max length of query is {0}".format(l))



if __name__ == '__main__':

  global_info_path = "/media/klaas/data/collection/disk12"
  relevance_params = {'res_file': os.path.join(global_info_path, 'desc.res'),
                      'qrels_file': os.path.join(global_info_path, 'qrels.clueweb09b.txt'),
                      'docnolist_file': os.path.join(global_info_path, 'docnolist'),
                      'output_file': os.path.join(global_info_path, 'relevance.clue.desc.fromres1000.pickle')}

  gov2_relevance_params = {"res_file": os.path.join(global_info_path, "title_BM25_0.4.res"),
                           "qrels_file": os.path.join(global_info_path, "qrels.merged"),
                           "docnolist_file": os.path.join(global_info_path, "docnolist"),
                           "output_file": os.path.join(global_info_path, "relevance.gov2.title.fromres1000.pickle")}

  wt10g_relevance_params = {"res_file": os.path.join(global_info_path, "BM25_T"),
                           "qrels_file": os.path.join(global_info_path, "qrels.trec9_10"),
                           "docnolist_file": os.path.join(global_info_path, "docnolist"),
                           "output_file": os.path.join(global_info_path, "relevance.wt10g.title.pickle")}

  disk12_relevance_params = {"res_file": os.path.join(global_info_path, "desc.BM25.c0.6.ed-1.et-1.res"),
                            "qrels_file": os.path.join(global_info_path, "qrels.trec123.adhoc"),
                            "docnolist_file": os.path.join(global_info_path, "docnolist.merge"),
                            "output_file": os.path.join(global_info_path, "relevance.disk12.desc.pickle")}
  # create_relevance(**disk12_relevance_params)
  # rel = load_pickle(disk12_relevance_params['output_file'])
  # print(rel.keys())
  # print(len(rel.keys()))
  #
  # rel1 = rel.get(101).get_supervised_docid_list()
  # print rel1
  # print(len(rel1))
  # print(rel.get(101).get_judged_docid_list())
  # print(sum(map(len, rel.get(101).get_judged_docid_list())))

  # pair_q_posnum = []
  # for key, relevance in rel.items():
  #   pair_q_posnum.append((key, sum(map(len, relevance.get_judged_docid_list()[1:]))))
  #
  # pair_q_posnum = sorted(pair_q_posnum, key=lambda x: x[1])
  # print(pair_q_posnum)
  # print(len(pair_q_posnum))
  # m, n = zip(*pair_q_posnum)
  # print(m)
  # print(n)
  # qid_list = np.ones((5, 30), dtype=np.int)
  # for i in range(30):
  #   l = np.random.permutation(m[5*i: 5*(i+1)])
  #   qid_list[:, i] = l
  #
  # qid_list = np.sort(qid_list)
  # print(qid_list.tolist())

  global_info_path = "/home/lcj/data/desc.disk12/features/global.info"
  idf_params = {'df_file': os.path.join(global_info_path, 'disk12.dfcf.txt'),
                'topic_file': os.path.join(global_info_path, 'disk12.desc.porter.morefilter.txt'),
                'output_file': os.path.join(global_info_path, 'desc.idf.pickle'),
                'maxlen': 24,
                'nb_doc': 741856}

  parse_idf_for_query(**idf_params)
  idf = load_pickle(idf_params['output_file'])
  print(idf)


  #
  # get_query_length("/media/klaas/data/collection/clue_final/topic/clueweb.desc.porter.txt")
  # get_query_length("/media/klaas/data/collection/disk12/disk12.desc.porter.morefilter.txt")
