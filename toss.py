import itertools

import numpy as np
from mcda_method import MCDA_method

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import distance_metrics as dists


class TOSS(MCDA_method):
    def __init__(self, normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean):
        """
        Create the TOSS method object and select normalization method `normalization_method` and
        distance metric `distance metric`.
        
        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`
        """
        self.normalization_method = normalization_method
        self.distance_metric = distance_metric


    def __call__(self, matrix, weights, types, s_coeff = 0):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives in rows and 
        n criteria in columns using criteria `weights` and criteria `types`.
        
        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            s_coeff: ndarray
                Vector with sustainability coefficient determined for each criterion
        
        Returns
        -------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 
        
        Examples
        ---------
        >>> toss = TOSS(normalization_method = minmax_normalization, distance_metric = euclidean)
        >>> pref = toss(matrix, weights, types, s_coeff)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        
        TOSS._verify_input_data(matrix, weights, types)
        return TOSS._toss(self, matrix, weights, types, self.normalization_method, self.distance_metric, s_coeff)


    # function for applying the SSP paradigm
    def _equalization(self, matrix, types, s_coeff):

        # Calculate mean deviation multiplied by s coefficient
        mad = (matrix - np.mean(matrix, axis = 0)) * s_coeff

        # Set as 0, those mean deviation values that for profit criteria are lower than 0
        # and those mean deviation values that for cost criteria are higher than 0
        for j, i in itertools.product(range(matrix.shape[1]), range(matrix.shape[0])):
            # for profit criteria
            if types[j] == 1:
                if mad[i, j] < 0:
                    mad[i, j] = 0
            # for cost criteria
            elif types[j] == -1:
                if mad[i, j] > 0:
                    mad[i, j] = 0

        # Subtract from performance values in decision matrix standard deviation values multiplied by a sustainability coefficient.
        return matrix - mad       


    @staticmethod
    def _toss(self, matrix, weights, types, normalization_method, distance_metric, s_coeff):
        # reducing compensation in normalized decision matrix
        e_matrix = self._equalization(matrix, types, s_coeff)

        # Normalize matrix using chosen normalization (for example linear normalization)
        n_matrix = normalization_method(e_matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = n_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance of every alternative from PIS and NIS using chosen distance metric `distance_metric` from `distance_metrics`
        Dp = np.array([distance_metric(x, pis) for x in weighted_matrix])
        Dm = np.array([distance_metric(x, nis) for x in weighted_matrix])

        C = Dm / (Dm + Dp)
        return C