from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import brent

class NormalApproximation(object):
    """Estimate popgen parameters using a normal approximation
    to the read data likelihood

    Arguments
    ---------
    ancient_dataset : AncientDataset
        ancient dataset object

    Attributes
    ----------
    n : int
        number of individuals
    p : int
        number of snps
    y : array
        n x p matrix of read data
    t : array
        n vector of times for each individual
    c : array
        n x p matrix of total coverage
    g : array
        n x p genotype matrix
    mu : array
        p vector of starting allele frequencies
    n_e : int
        effective population size
    m : array
        n x n matrix storing min times
    sigma_m : array
        n x n matrix storing component of covariance
        precomputed as its reused in optimization
    opt_res : minimize object
        object returned after minimizing the negative log likelihood
    """
    def __init__(self, ancient_dataset):

        # ancient dataset object!
        self.ancient_dataset = ancient_dataset

        # number of individuals
        self.n = ancient_dataset.n

        # number of snps
        self.p = ancient_dataset.p

        # n x p matrix of read data
        self.y = ancient_dataset.y

        # n vector of times
        self.t = ancient_dataset.t

        # n x p matrix of total coverage
        self.c = ancient_dataset.c

        # n x p matrix of genotypes
        self.g = ancient_dataset.g

        # p vector of starting allele frequencies
        self.mu = ancient_dataset.mu

        # effective population size
        self.n_e = ancient_dataset.n_e

        # n x n matrix storing min times for each pair including self pairs
        self.T = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.T[i, j] = np.min(np.array([self.t[i], self.t[j]]))

        self.T_diag = np.diag(np.diag(self.T))

    def fit(self):
        """Estimate n_e via maxium likelihood in this case we are starting
        by fixing mu and g
        """
        self.opt_res = minimize(self._neg_log_lik, np.array([0.0]))

    def _neg_log_lik(self, z_e):
        """Negative log likelihood of the full dataset

        Arguments
        ---------
        z_e : float
            log effective population size

        Returns
        -------
        nll : float
            negative log likelihood
        """
        # transform back to n_e
        n_e = np.exp(z_e)

        # log likelihood
        ll = 0.0

        # loop over the snps
        for j in range(self.p):
            ll += self._snp_log_lik(j, n_e)

        # minus log likelihood
        nll = -ll
        return(nll)

    def _snp_log_lik(self, j, n_e):
        """Likelihood for a single snp

        Arguments
        ---------
        j : int
            index of jth snp
        n_e : float
            effective population size

        Returns
        -------
        l_j : float
            likelihood for a single snp
        """
        # coal rate
        q = 1.0 / (2.0 * n_e)

        # heterozygosity at snp j
        h_j = 2.0 * self.mu[j] * (1.0 - self.mu[j])

        #
        tau_j = .5 * q * h_j

        # first component of variance covaraince matrix
        sigma_j0 = (((self.c[:, j] @ self.c[:, j].T) / 4.0) *
                    (tau_j * ((4. * self.T) - (2. * self.T_diag)) + (h_j * np.eye(self.n)))
                   )

        # second component of variance covaraince matrix
        sigma_j1 = ((np.diag(self.c[:, j]) / 2.0) *
                    (.5 * h_j * np.eye(self.n)) - (tau_j * self.T_diag)
                   )

        # variance covariance matrix
        sigma_j = sigma_j0 + sigma_j1

        print(sigma_j)

        # likelhood of jth snp
        ll_j = stats.multivariate_normal.logpdf(x=self.y[:, j], mean=self.mu[j] * self.c[:, j], cov=sigma_j)

        return(ll_j)
