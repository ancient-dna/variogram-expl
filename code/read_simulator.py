from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class AncientDataset(object):
    """Simulate read data with genotypes generated
    under a simple Wright-Fisher Markov Chain

    Arguments
    ---------
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

    max_time : int
        the max time point for simulating an individual

    eps : float
        sequencing error rate

    Attributes
    ----------
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

    max_time : int
            the max time point for simulating an individual
    """
    def __init__(self, n, p, alpha_f=.5, beta_f=.5,
                 n_e=10000, mean_cov=5, n_gen=2000,
                 eps=1e-3):

        # number of inds
        self.n = n

        # number of snps
        self.p = p

        # hyper params of starting freq beta dist
        self.alpha_f = alpha_f
        self.beta_f = beta_f

        # effective population size
        self.n_e = n_e

        # average coverage
        self.mean_cov = mean_cov

        # number of generations to sim
        self.n_gen = n_gen

        # simulate times for each individual
        self.t = np.sort(np.random.randint(0., max_time, n))

        # simulate starting allele frequencies for each snp
        self.mu = np.random.beta(alpha_f, beta_f, p)

    def gen_freq(self):
        """Generate allele frequencies
        under the Wright-Fisher Markov Chain
        """
        self.f = np.empty([self.n_gen, self.p])
        self.f[0, :] = self.mu
        for t in range(1, self.n_gen):
            self.f[t, :] = np.random.binomial(self.n_e, self.f[t-1, :]) / self.n_e

    def gen_geno(self):
        """Simulate genotypes assuming hardy
        weinberg equilibrim
        """
        ones = np.ones([self.n, self.p], dtype=np.int64)
        self.g = np.random.binomial(2 * ones, self.f[self.t, :])

    def gen_reads(self):
        """Simulate the count of alt reads
        """
        # n x p matrix of coverage for each ind i and site j
        self.c = np.random.poisson(self.mean_cov, [self.n, self.p])

        # simulate with some small error prob
        q = self.g / 2.
        pi = ((1. - self.eps) * q) + (self.eps * (1. - q))

        # n x p matrix count of counted alleles
        self.y = np.random.binomial(self.c, pi)
