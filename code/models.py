from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class NormalApproximation(object):
    """Estimate popgen parameters using a normal approximation
    to the read data likelihood
    """
    def __init__(self, ancient_dataset):

        self.ancient_dataset = ancient_dataset

        # n x n matrix storing min times for each pair including self pairs
        self.m = np.empty((self.ancient_dataset.n, self.ancient_dataset.n))
        for i in range(self.ancient_dataset.n):
            for j in range(self.ancient_dataset.n):
                self.m[i, j] = np.min(np.array([self.ancient_dataset.t[i],
                                                self.ancient_dataset.t[j]]))



    def _log_likelihood(self, n_e):
        """
        """
        diag_idx = np.diag_indices(self.ancient_dataset.n, ndim=2)

    def _snp_likelihood(self, y_j, c_j, mu_j, g_j, n_e):
        """
        """
        pass
