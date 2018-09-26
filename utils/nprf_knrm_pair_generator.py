
import os
import numpy as np

from relevance_info import Relevance
from pair_generator import PairGenerator

class NPRFKNRMPairGenerator(PairGenerator):

  def __init__(self, relevance_dict_path,
               dd_d_feature_path,  sample_perquery_limit,
               sample_total_limit,
               doc_topk_term=30, nb_supervised_doc=20, kernel_size=30, batch_size=32, shuffle=True):

    super(NPRFKNRMPairGenerator, self).__init__(relevance_dict_path, batch_size, shuffle,
                                                sample_perquery_limit, sample_total_limit)

    # self.query_maxlen = query_maxlen
    self.doc_topk_term = doc_topk_term
    self.nb_supervised_doc = nb_supervised_doc
    self.kernel_size = kernel_size

    self.dd_d_feature_path = dd_d_feature_path

  def generate_pair_batch(self, qid_list, sample_size):

    nb_pairs = self.count_pairs_balanced(qid_list, sample_size)
    nb_batches = nb_pairs / self.batch_size

    triplet_list = self.get_triplet_list_balanced(qid_list, sample_size)

    if self.shuffle == True:
      triplet_list = np.random.permutation(triplet_list)

    while True:
      for i in range(nb_batches):
        curr_triplet_list = triplet_list[i * self.batch_size: (i + 1) * self.batch_size]
        dd_d, Y = self.get_feature_batch(curr_triplet_list)
        score_gate = self.get_baseline_score(curr_triplet_list)
        yield [dd_d, score_gate], Y


  def get_feature_batch(self, triplet_list):

    # qd_d_feature = np.zeros((self.batch_size * 2, self.kernel_size), dtype=np.float32)
    dd_d_feature = np.zeros((self.batch_size * 2, self.nb_supervised_doc, self.kernel_size), dtype=np.float32)

    Y = np.ones(self.batch_size * 2)

    for i, (qid, docid_neg, docid_pos) in enumerate(triplet_list):

      '''
      # qxd
      qd_d_neg_file = os.path.join(self.qd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid_neg))
      qd_d_pos_file = os.path.join(self.qd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid_pos))

      # interleavef format: neg, positive
      qd_d_feature[2 * i, :] = np.load(qd_d_neg_file)
      qd_d_feature[2 * i + 1, :] = np.load(qd_d_pos_file)
      '''

      # dxd
      dd_d_neg_file = os.path.join(self.dd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid_neg))
      dd_d_pos_file = os.path.join(self.dd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid_pos))

      dd_d_neg_array = np.load(dd_d_neg_file)
      dd_d_pos_array = np.load(dd_d_pos_file)
      dd_d_neg_all = np.zeros((self.nb_supervised_doc, self.doc_topk_term, self.kernel_size), dtype=np.float32)
      dd_d_pos_all = np.zeros((self.nb_supervised_doc, self.doc_topk_term, self.kernel_size), dtype=np.float32)
      for j in range(self.nb_supervised_doc):
        dd_d_neg_all[j][: len(dd_d_neg_array[j])] = dd_d_neg_array[j][: self.doc_topk_term]
        dd_d_pos_all[j][: len(dd_d_pos_array[j])] = dd_d_pos_array[j][: self.doc_topk_term]

      dd_d_feature[2 * i, :] = np.sum(dd_d_neg_all, axis=1)
      dd_d_feature[2 * i + 1, :] = np.sum(dd_d_pos_all, axis=1)


    return dd_d_feature, Y

  def get_baseline_score(self, triplet_list):
    score_gate = np.zeros((self.batch_size * 2, self.nb_supervised_doc, 1), dtype=np.float32)
    for i, (qid, docid_neg, docid_pos) in enumerate(triplet_list):
      relevance = self.relevance_dict.get(qid)
      score_list = relevance.get_supervised_score_list()[: self.nb_supervised_doc]
      max_score, min_score = score_list[0], score_list[-1]
      selected_score = np.asarray(score_list, dtype=np.float32)
      selected_score = 0.5 * (selected_score - min_score) / (max_score - min_score) + 0.5
      #selected_score = np.asarray(score_list, dtype=np.float32) + max_score
      selected_score = selected_score.reshape((self.nb_supervised_doc, 1))
      score_gate[2 * i, :], score_gate[2 * i + 1, :] = selected_score, selected_score

    return score_gate

  def list_batch_perquery(self, qid, topk):

    relevance = self.relevance_dict.get(qid)
    supervised_docid_list = relevance.get_supervised_docid_list()
    rerank_docid_list = supervised_docid_list[: topk]
    len_indicator = len(rerank_docid_list)
    # qd_d = np.zeros((len_indicator, self.kernel_size), dtype=np.float32)
    dd_d = np.zeros((len_indicator, self.nb_supervised_doc, self.kernel_size), dtype=np.float32)

    # for score
    topk_score_list = relevance.get_supervised_score_list()[: self.nb_supervised_doc]
    max_score, min_score = topk_score_list[0], topk_score_list[-1]
    topk_score_list = np.asarray(topk_score_list, dtype=np.float32)
    topk_score_list = 0.5 * (topk_score_list - min_score) / (max_score - min_score) + 0.5
    score = np.tile(topk_score_list, len_indicator)
    score = score.astype(np.float32).reshape((len_indicator, self.nb_supervised_doc, 1))

    # for document
    for i, docid in enumerate(rerank_docid_list):
      # qd_d_file = os.path.join(self.qd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid))
      dd_d_file = os.path.join(self.dd_d_feature_path, str(qid), "q{0}_d{1}.npy".format(qid, docid))
      # qd_d[i] = np.load(qd_d_file)
      dd_d_array = np.load(dd_d_file)
      dd_d_all = np.zeros((self.nb_supervised_doc, self.doc_topk_term, self.kernel_size), dtype=np.float32)
      for j in range(self.nb_supervised_doc):
        dd_d_all[j][: len(dd_d_array[j])] = dd_d_array[j][: self.doc_topk_term] 

      dd_d[i] = np.sum(dd_d_all, axis=1)

    return dd_d, score, len_indicator

  def generate_list_batch(self, qid_list, topk):

    dd_d_all, score_all, len_indicator = self.list_batch_perquery(qid_list[0], topk)
    len_indicator_all = [len_indicator]
    for qid in qid_list[1:]:
      dd_d, score, len_indicator = self.list_batch_perquery(qid, topk)
      # qd_d_all = np.concatenate((qd_d_all, qd_d), axis=0)
      dd_d_all = np.concatenate((dd_d_all, dd_d), axis=0)
      score_all = np.concatenate((score_all, score), axis=0)
      len_indicator_all.append(len_indicator)

    return dd_d_all, score_all, len_indicator_all


  def get_triplet_list(self, qid_list, sample_size=10):

    triplet_list_global = []
    for qid in qid_list:
      relevance = self.relevance_dict.get(qid)
      relevance_posting = relevance.get_judged_docid_list()
      res = relevance.get_supervised_docid_list()
      if len(res) < self.nb_supervised_doc:
        pass # because that d2d feature cannot be constructed
      else:
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


  def count_pairs(self, qid_list, sample_size):

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
      res = relevance.get_supervised_docid_list()
      if len(res) < self.nb_supervised_doc:
        pass # because d2d feature cannnot be constructed
      else:
        rel_0, rel_1, rel_2 = relevance_posting[0], relevance_posting[1], relevance_posting[2]

        count_01 = count_on_topic(len(rel_0), len(rel_1), sample_size)
        count_12 = count_on_topic(len(rel_1), len(rel_2), sample_size)
        count = min(self.sample_perquery_limit, count_01 + count_12)
        total += count
    total = min(self.sample_total_limit, total)

    return total
