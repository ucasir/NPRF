
import os
import re
import sys
import subprocess
from scipy import stats

parent_path = '/home/lcj/tool/IREvaluation'
trec_eval_script_path = os.path.join(parent_path, 'trec_eval.9.0/trec_eval')


def run(command, get_ouput=False):
  try:
    if get_ouput:
      process = subprocess.Popen(command, stdout=subprocess.PIPE)
      output, err = process.communicate()
      exit_code = process.wait()
      return output
    else:
      subprocess.call(command)
  except subprocess.CalledProcessError as e:
    print e


def evaluate_trec(qrels, res):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)

  MAP = re.findall(r'map\s+all.+\d+', output)[0].split('\t')[2].strip()
  P20 = re.findall(r'P_20\s+all.+\d+', output)[0].split('\t')[2].strip()
  NDCG20 = re.findall(r'ndcg_cut_20\s+all.+\d+', output)[0].split('\t')[2].strip()
  # print gd_output.split('\n')[-2]
  # NDCG20, ERR20 = gd_output.split('\n')[-2].split(',')[2:4]

  return MAP, P20, NDCG20


def evaluate_trec_per_query(qrels, res):

  command = [trec_eval_script_path, '-q', qrels, res]
  output = run(command, get_ouput=True)
  gd_command = [gd_eval_script_path, qrels, res] #+ " | awk -F',' '{print $3}'"
  gd_output = run(gd_command, get_ouput=True)

  MAP_set = re.findall(r'map\s+\t\d+.+\d+', output)
  P20_set = re.findall(r'P_20\s+\t\d+.+\d+', output)

  NDCG20_set, ERR20_set = [], []
  for line in gd_output.split('\n')[1: -2]:
    ndcg, err = line.split(',')[2: 4]
    NDCG20_set.append(float(ndcg))
    ERR20_set.append(float(err))

  #print len(NDCG20_set)
  #print NDCG20_set
  MAP_set = map(lambda x: float(x.split('\t')[-1]), MAP_set)
  P20_set = map(lambda x: float(x.split('\t')[-1]), P20_set)


  return MAP_set, P20_set, NDCG20_set

def tt_test(qrels, res1, res2):
  MAP_set1, P20_set1, NDCG20_set1 = evaluate_trec_per_query(qrels, res1)
  MAP_set2, P20_set2, NDCG20_set2 = evaluate_trec_per_query(qrels, res2)
  '''
  print(P20_set1)
  print(P20_set2)
  print(NDCG20_set1)
  print(NDCG20_set2)
  print(len([t for t in np.asarray(MAP_set2) - np.asarray(MAP_set1) if t > 0]))
  print(len([t for t in np.asarray(P20_set2) - np.asarray(P20_set1) if t > 0]))
  print(len([t for t in np.asarray(NDCG20_set2) - np.asarray(NDCG20_set1) if t > 0]))
  '''
  t_value_map, p_value_map = stats.ttest_rel(MAP_set1, MAP_set2)
  t_value_p20, p_value_p20 = stats.ttest_rel(P20_set1, P20_set2)
  t_value_ndcg20, p_value_ndcg20 = stats.ttest_rel(NDCG20_set1, NDCG20_set2)

  return p_value_map, p_value_p20, p_value_ndcg20


if __name__ == '__main__':
  qrels = '/home/lcj/data/robust04/original/qrels.robust2004'
  res = '/media/klaas/research/01_ir/BM25RocQEBase/robust_docnos.res'
  #print evaluate_trec(qrels, res,)
  argv = sys.argv
  res1, res2 = argv[1], argv[2]
  print(tt_test(qrels, res1, res2))
