# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython.parallel import prange
from libc.math cimport fabs


def prediction_errors(double[::1] errors, double[::1] ratings, double[::1] ratings_inv, double[::1] scored, int[::1] opponents, int[::1] opp_played, int[::1] indices, int n_players):
    cdef int j, k
    cdef double score

    for j in prange(n_players, nogil=True):
        score = 0.0
        for k in range(indices[j], indices[j + 1]):
            score = score + opp_played[k] / (1.0 + (ratings[opponents[k]] * ratings_inv[j]))
        errors[j] = scored[j] - score


def adjust(double[::1] ratings, double[::1] errors, double[::1] total_inv, int n_players, double K):
    cdef int i
    
    for i in prange(n_players, nogil=True):
        ratings[i] = ratings[i] * 10 ** (K * errors[i] * total_inv[i])
