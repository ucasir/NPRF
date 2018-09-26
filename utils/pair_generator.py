
import numpy as np
import cPickle as pickle

import os
import sys
import math
import logging
sys.path.append('../utils')

from relevance_info import Relevance

class PairGenerator(object):
  '''Generator used for neg/pos pair generation, follow the interleaved format

  Attributes:
      relevance_dict_path: path to the whole relevance information
      batch_size: number of samples per step
      shuffle: whether to shuffle
      sample_perquery_limit: limit on each query
      sample_total_limit: limit on one epoch

  '''
  def __init__(self, relevance_dict_path, batch_size, shuffle, sample_perquery_limit, sample_total_limit):

    self.batch_size = batch_size
    self.shuffle = shuffle
    self.sample_perquery_limit = sample_perquery_limit
    self.sample_total_limit = sample_total_limit

    with open(relevance_dict_path, 'r') as f:
      self.relevance_dict = pickle.load(f)


  def generate_pair_batch(self, qid_list, sample_size):
    '''

    Args:
      qid_list: the training qid list
      sample_size: number of negative samples accompanied with positive sample

    Returns:
      A generator that yield (feature, label) pairs.
    '''
    pass


  def get_feature_batch(self, triplet_list):
    '''

    Args:
      triplet_list: list of (qid, neg_docid, pos_docid) triplets

    Returns:
      feature for current batch
    '''
    pass


  def generate_list_batch(self, qid_list, topk):
    '''

    Args:
      qid_list: the validation/test qid list
      topk: the number of top document to be re-ranked

    Returns:
      features, plus the length indicator.
    '''
    pass


  def get_triplet_list(self, qid_list, sample_size=10):
    '''Deprecated, please use get_triplet_list_balanced

    '''
    triplet_list_global = []
    for qid in qid_list:
      relevance = self.relevance_dict.get(qid)
      relevance_posting = relevance.get_judged_docid_list()
      rel_0, rel_1, rel_2 = relevance_posting[0], relevance_posting[1], relevance_posting[2]
      rel_01_triplet_list = self.create_triplet_list(rel_0, rel_1, qid, sample_size)
      rel_12_triplet_list = self.create_triplet_list(rel_1, rel_2, qid, sample_size)

      curr_triplet_list = []
      if rel_01_triplet_list != None:
        curr_triplet_list.extend(rel_01_triplet_list)
      if rel_12_triplet_list != None:
        curr_triplet_list.extend(rel_12_triplet_list)
      curr_triplet_list = np.random.permutation(curr_triplet_list)
      triplet_list_global.extend(curr_triplet_list[: self.sample_perquery_limit])

    triplet_list_global = np.random.permutation(triplet_list_global)
    triplet_list_global = triplet_list_global[: self.sample_total_limit]

    return triplet_list_global



  def get_triplet_list_balanced(self, qid_list, sample_size):
    '''

    Args:
      qid_list:
      sample_size: number of sampled negative document for each positive document

    Returns:

    '''
    triplet_list_global = []
    for qid in qid_list:
      relevance = self.relevance_dict.get(qid)
      relevance_posting = relevance.get_judged_docid_list()
      pos_list = []
      for i in range(len(relevance_posting) - 1, 0, -1):
        pos_list += relevance_posting[i]
      num_of_positive = len(pos_list)
      curr_sample_size, curr_perquery_limit = self._decide_sample(sample_size, num_of_positive)

      curr_triplet_list = []
      for i in range(len(relevance_posting) - 1, 0, -1):
        curr_positive_docid_list = relevance_posting[i]
        curr_negative_docid_list = []
        for j in range(i - 1, -1, -1):
          curr_negative_docid_list.extend(relevance_posting[j])
        pair_list =  self.create_triplet_list(curr_negative_docid_list, curr_positive_docid_list, qid, curr_sample_size)
        if pair_list != None:
          curr_triplet_list.extend(pair_list)
      curr_triplet_list = np.random.permutation(curr_triplet_list)[: curr_perquery_limit]
      triplet_list_global.extend(curr_triplet_list)
    triplet_list_global = np.random.permutation(triplet_list_global)
    logging.info("Generate totally {0} pairs".format(len(triplet_list_global)))
    return triplet_list_global


  def create_triplet_list(self, neg_docid_list, pos_docid_list, qid, sample_size):
    '''create triplet list from negative docid list and positive docid list

        Strategy: for each docid in pos_docid_list, sample negative docids from neg_docid_list
        with the given sample size.

    Args:
      neg_docid_list:
      pos_docid_list:
      qid:
      sample_size:

    Returns:

    '''
    sample_size = min(len(neg_docid_list), sample_size)
    if sample_size == 0:
      return
    triplet_list = []
    for pos_docid in pos_docid_list:
      neg_sample_list = np.random.choice(neg_docid_list, sample_size, replace=False)
      pos_sample_list = [pos_docid] * sample_size
      triplet_list.extend(zip([qid] * sample_size, neg_sample_list, pos_sample_list))

    return  triplet_list

  def count_pairs(self, qid_list, sample_size):
    '''Deprecated,  please use func count_pairs_balanced

    Args:
      qid_list:
      sample_size:

    Returns:

    '''
    def count_on_topic(neg_len, pos_len, sample_size):
      sample_size = min(neg_len, sample_size)
      if sample_size == 0:
        return 0
      else:
        return pos_len * sample_size

    total = 0
    for qid in qid_list:
      relevance = self.relevance_dict.get(qid)
      relevance_posting = relevance.get_judged_docid_list()
      rel_0, rel_1, rel_2 = relevance_posting[0], relevance_posting[1], relevance_posting[2]

      count_01 = count_on_topic(len(rel_0), len(rel_1), sample_size)
      count_12 = count_on_topic(len(rel_1), len(rel_2), sample_size)
      count = min(self.sample_perquery_limit, count_01 + count_12)
      total += count
    total = min(self.sample_total_limit, total)

    return total

  def count_pairs_balanced(self, qid_list, sample_size):
    '''Count the number of total pair created in each epoch, which could be used as a keras function parameter

    Args:
      qid_list:
      sample_size:

    Returns:

    '''
    def count_on_topic(neg_len, pos_len, sample_size):
      sample_size = min(neg_len, sample_size)
      if sample_size == 0:
        return 0
      else:
        return pos_len * sample_size

    total = 0
    for qid in qid_list:
      relevance = self.relevance_dict.get(qid)
      relevance_posting = relevance.get_judged_docid_list()
      pos_list = []
      for i in range(len(relevance_posting) - 1, 0, -1):
        pos_list += relevance_posting[i]
      num_of_positive = len(pos_list)
      curr_sample_size, curr_perquery_limit = self._decide_sample(sample_size, num_of_positive)

      count_perquery = 0
      for i in range(len(relevance_posting) - 1, 0, -1):
        curr_positive_docid_list = relevance_posting[i]
        curr_negative_docid_list = []
        for j in range(i - 1, -1, -1):
          curr_negative_docid_list.extend(relevance_posting[j])
        count_perquery += count_on_topic(len(curr_negative_docid_list), len(curr_positive_docid_list), curr_sample_size)
      count_perquery = min(count_perquery, curr_perquery_limit)
      total += count_perquery

    return total

  def count_percentile(self):
    ''' cut number of into percetiles, sample size of query that has more positive documents will be cut

    Returns:

    '''
    num_pos_list = []
    for qid, relevance in self.relevance_dict.items():
      judged_docid_list = relevance.get_judged_docid_list()
      rel_docid_list = []
      for i in range(len(judged_docid_list) - 1, 0, -1):
        rel_docid_list += judged_docid_list[i]
      num_pos_list.append(len(rel_docid_list))
    num_pos_list = sorted(num_pos_list)
    # percentile of 1/4 2/4 3/4
    return [num_pos_list[len(num_pos_list) * i / 4] for i in range(1, 4)]

  def _decide_sample(self, sample_size, num_of_positive):
    '''

    Args:
      sample_size:
      num_of_positive: number of positive documents in current query

    Returns:
      modified sample size.
    '''
    percentile_14, percentile_24, percentile_34 = self.count_percentile()
    curr_sample_size, curr_perquery_limit = sample_size, self.sample_perquery_limit
    if num_of_positive < percentile_14:
      curr_sample_size, curr_perquery_limit = curr_sample_size * 5, curr_perquery_limit * 5
    elif num_of_positive < percentile_24:
      curr_sample_size, curr_perquery_limit = curr_sample_size * 3, curr_perquery_limit * 3
    elif num_of_positive < percentile_34:
      curr_sample_size, curr_perquery_limit = curr_sample_size * 1.5, curr_perquery_limit * 1.5

    return int(math.floor(curr_sample_size)), int(math.floor(curr_perquery_limit))


