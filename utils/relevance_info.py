
class Relevance(object):
  '''Incooperation with result file and qrels file

  Attributes:
      qid, int, query id
      judgement_docid_list: list of list, judged docids in TREC qrels
      supervised_docid_list: list, top docids from unsupervised models, e.g. BM25, QL.

  '''
  def __init__(self, qid, judged_docid_list, supervised_docid_list, supervised_score_list):

    self._qid = qid
    self._judged_docid_list = judged_docid_list
    self._supervised_docid_list = supervised_docid_list
    self._supervised_score_list = supervised_score_list

  def get_qid(self):
    return self._qid

  def get_judged_docid_list(self):
    return self._judged_docid_list

  def get_supervised_docid_list(self):
    return self._supervised_docid_list

  def get_supervised_score_list(self):
    return self._supervised_score_list