from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import pickle as pkl
import os

class ReadSimulator(object):
    """Simulate read data under genotypes generated
    under a simple Wright-Fisher Markov Chain

    Arguments:
        n : int
            number of individuals
        p : int
            number of snps
        alpha_f : float
            hyperparameter of beta distribution for
            simulating starting allele frequencies
        beta_f : float
            hyperparameter of beta distribution for
            simulating starting allele frequencies
        n_e : int
            effective population size
        mean_cov : int
            average coverage to simulate each site
        max_time : float
            the max time point for simulating an individual
    """
    def __init__(self, n, p, alpha_f=.5, beta_f=.5, n_e=10000,
                 mean_cov=5, max_time=5000):
        self.alpha_f = alpha_f
        self.beta_f = beta_f
        self.n_e = n_e
        self.mean_cov = mean_cov
        self.max_time = max_time
        # simulate times for each individual
        self.t = np.sort(np.random.uniform(0., max_time, n))
        # simulate starting allele frequencies for each snp
        self.mu = np.random.beta(alpha_f, beta_f, p)

    def gen_f(self):
        """Generate allele frequencies
        under the Wright-Fisher Markov Chain
        """
        pass
