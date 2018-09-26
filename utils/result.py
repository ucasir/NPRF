

class Result(object):
  '''Ease write/read operation on standard TREC result file

  Attributes:
      qid:
      docid_list: docid (not docno) list from current qid
      score_list: scores mapping to docids
      runid: unique run id for current run.
  '''
  def __init__(self, qid, docid_list, score_list, runid):
    self._qid = qid
    self._docid_list = docid_list
    self._score_list = score_list
    self._runid = runid

  def get_qid(self):
    return self._qid

  def get_docid_list(self):
    return self._docid_list

  def get_score_list(self):
    return self._score_list

  def get_runid(self):
    return self._runid

  def update_ranking(self):
    pair = zip(self._docid_list, self._score_list)
    updated_pair = sorted(pair, key=lambda x: x[1], reverse=True)
    self._docid_list, self._score_list = zip(*updated_pair)

  def set_docid_list(self, docid_list):
    self._docid_list = docid_list

  def set_score_list(self, score_list):
    self._score_list = score_list

  def update_all(self, docid_list, score_list):
    self.set_docid_list(docid_list)
    self.set_score_list(score_list)
    self.update_ranking()

    