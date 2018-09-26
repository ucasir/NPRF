
import numpy as np

from bs4 import BeautifulSoup
from gensim.models.keyedvectors import KeyedVectors
from gensim.matutils import unitvec, cossim

def get_word_vec(word, embeddings, OOV_dict=None):

  vector_size = embeddings.vector_size
  try:
    vec = embeddings.word_vec(word)
  except KeyError:
    vec = OOV_dict.get(word)
    if vec is None:
      vec = np.random.uniform(low=-1., high=1., size=vector_size)
      OOV_dict.update({word: vec})
      #warnings.warn(word + " not in vocabulary")

  return vec


def similarity_matrix(query, doc, embeddings, OOV_dict):

  vector_size = embeddings.vector_size
  q_mat = np.zeros((len(query), vector_size))#, dtype=np.float32)
  d_mat = np.zeros((vector_size, len(doc)))#, dtype=np.float32)

  for i, word in enumerate(query):
    q_mat[i, :] = unitvec(get_word_vec(word, embeddings, OOV_dict))
  for j, word in enumerate(doc):
    d_mat[:, j] = unitvec(get_word_vec(word, embeddings, OOV_dict))
  similarity_matrix = np.dot(q_mat, d_mat)
  #similarity_matrix = similarity_matrix.astype(np.float)

  return similarity_matrix


def hist_from_matrix(text_maxlen, hist_size, sim_mat):
  '''

    References: https://github.com/faneshion/MatchZoo/blob/master/matchzoo/inputs/preprocess.py#L425
  Args:
    text_maxlen:
    hist_size:
    sim_mat:

  Returns:

  '''
  hist = np.zeros((text_maxlen, hist_size), dtype=np.float32)
  # mm = sim_mat
  for (i, j), v in np.ndenumerate(sim_mat):
    if i >= text_maxlen:
      break
    vid = int((v + 1.) / 2. * (hist_size - 1.))
    hist[i][vid] += 1
  hist += 1
  hist = np.log10(hist)
  # yield hist
  return hist


def kernel_from_matrix(sim_mat, mu_list, sigma_list, d2d=False):
  '''

    References: https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_knrm.py#L74
  Args:
    sim_mat:
    mu_list:
    sigma_list:
    d2d:

  Returns:

  '''
  assert len(mu_list) == len(sigma_list)
  text1_len = sim_mat.shape[0]
  if d2d:
    kernel_feature = np.zeros((text1_len, len(mu_list)), dtype=np.float32)
  else:
    kernel_feature = np.zeros((len(mu_list),), dtype=np.float32)
  for i in range(len(mu_list)):
    mu = mu_list[i]
    sigma = sigma_list[i]
    tmp = np.exp(-np.square(sim_mat - mu) / (2 * np.square(sigma)))  # RBF kernel
    kde = np.sum(tmp, axis=1)
    kde = np.log(np.maximum(kde, 1e-10)) * 0.01
    if d2d:
      kernel_feature[:, i] = kde
    else:
      kernel_feature[i] = (np.sum(kde))

  return kernel_feature


def kernal_mus(n_kernels, use_exact):
  """
  get the mu for each guassian kernel. Mu is the middle of each bin
  :param n_kernels: number of kernels (including exact match). first one is exact match
  :return: l_mu, a list of mu.
  """
  if use_exact:
    l_mu = [1]
  else:
    l_mu = [2]
  if n_kernels == 1:
    return l_mu

  bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
  l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
  for i in xrange(1, n_kernels - 1):
    l_mu.append(l_mu[i] - bin_size)
  return l_mu


def kernel_sigmas(n_kernels, lamb, use_exact):
  """
  get sigmas for each guassian kernel.
  :param n_kernels: number of kernels (including exactmath.)
  :param lamb:
  :param use_exact:
  :return: l_sigma, a list of simga
  """
  bin_size = 2.0 / (n_kernels - 1)
  l_sigma = [0.00001]  # for exact match. small variance -> exact match
  if n_kernels == 1:
    return l_sigma

  l_sigma += [bin_size * lamb] * (n_kernels - 1)
  return l_sigma