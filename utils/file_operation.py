# -*- coding: UTF-8 -*-

import os
import gzip
import time
import glob
import logging
import numpy as np
import cPickle as pickle

from result import Result
from bs4 import BeautifulSoup
from collections import OrderedDict
from relevance_info import Relevance

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

########################### evaluation oriented #######################
def write_result_to_trec_format(result_dict, write_path, docnolist_file):

  docnolist = parse_corpus(docnolist_file)
  f = open(write_path, 'w')
  for qid, result in result_dict.items():

    docid_list = result.get_docid_list()
    score_list = result.get_score_list()
    rank = 0
    for docid, score in zip(docid_list, score_list):
      f.write("{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(qid, docnolist[docid], rank, score, result.get_runid()))
      rank += 1

  f.close()


def read_result_from_file(result_file, docnolist_file):
  docnolist = parse_corpus(docnolist_file)
  docid_map = dict(zip(docnolist, range(len(docnolist))))

  res_all = parse_corpus(result_file)
  res_dict = OrderedDict()
  prev_qid, runid = -1, -1
  docid_list = []
  score_list = []
  for line in res_all:
    tokens = line.split()
    qid, docid, score, runid = int(tokens[0]), docid_map.get(tokens[2]), float(tokens[4]), tokens[5]
    if qid != prev_qid:
      if len(docid_list) > 0:
        result = Result(qid, docid_list, score_list, runid)
        res_dict.update({prev_qid: result})
      docid_list, score_list = [docid], [score]
      prev_qid = qid
    else:
      docid_list.append(docid)
      score_list.append(score)
  res = Result(prev_qid, docid_list, score_list, runid)
  res_dict.update({prev_qid: res})

  return res_dict

############################## corpus #################################

def representsInt(s):
  try:
    int(s)
    return True
  except ValueError:
    return False


def docno2docid(docno_file, increment=0):
  docno_map = {}
  docid = increment
  with open(docno_file, 'r') as f:
    for line in f:
      docno_map.update({line.strip(): docid})
      docid += 1

  return docno_map


def parse_topic(topic_file):
  query_dict = {}
  with open(topic_file, 'r') as f:
    for line in f:  # .readlines():
      tokens = line.strip().split()
      qid = tokens[0].strip()
      q_terms = tokens[1:]
      query_dict.update({int(qid): q_terms})

  return query_dict

def parse_stoplist(stop_file):
  stoplist = []
  with open(stop_file, 'r') as f:
    _file = f.read()
  stop_soup = BeautifulSoup(_file, 'lxml')
  for a in stop_soup.find_all('word'):
    stoplist.append(a.text)

  return stoplist

def df_map_from_file(df_file):
  df_map = {}
  with open(df_file, 'r') as f:
    for line in f:
      tokens = line.strip().split('\t')
      term, df, cf = tokens[0], int(tokens[1]), int(tokens[2])
      if df_map.has_key(term): # this is due to that two terms maybe the same after Krovetz stemming
        df_map[term] += df
      else:
        df_map.update({term: df})

  return df_map


def parse_content(content, stoplist=None):
  '''format: docno\tdoclen\tterm:count
  '''
  docno, doclen, raw_content = content.strip().split('\t')
  term_count_pair =  list(map(lambda x: x.split(':'), raw_content.split()))
  content_list = []
  for term, count in term_count_pair:
    count = int(count)
    if (stoplist != None and term in stoplist) or representsInt(term) or len(term) < 2:
      pass
    else:
      content_list.extend([term] * count)
  #assert int(doclen) == len(content_list)
  #string = ' '.join(content_list)

  return content_list


def parse_topk_content(content):

  try:
    docno, doclen, terms = content.strip().split('\t')
    terms = terms.split()
    return terms
  except ValueError, e:
    print(e)
    print(content)
    return []

def parse_corpus(corpus_file):
  corpus = []
  with open(corpus_file, 'r') as f:
    for line in f:
      corpus.append(line.strip())

  return corpus


########################### I/O operation #######################

def retain_file(path, tagger, retain_file):
  files = glob.glob(os.path.join(path, tagger + '*'))
  for _file in files:
    if not _file.endswith(retain_file):
      os.remove(_file)


def make_directory(path1, path2):
  dir = os.path.join(path1, path2)
  if not os.path.exists(dir):
    try:
      os.mkdir(dir)
    except OSError, e:
      print(e)
      time.sleep(10)

  return dir


def load_pickle_and_gzip(file_name):
  with gzip.open(file_name, 'rb') as f:
    res = pickle.load(f)
  return res


def save_pickle_and_gzip(obj, file_name):
  with gzip.open(file_name, 'wb') as f:
    pickle.dump(obj, f)


def load_pickle(_file):
  with open(_file, 'r') as f:
    res = pickle.load(f)
  return res


def save_pickle(obj, _file):
  with open(_file, 'wb') as f:
    pickle.dump(obj, f)