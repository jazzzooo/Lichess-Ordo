# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython.parallel import prange


def prediction_errors(double[::1] errors, double[::1] ratings, double[::1] scored, int[::1] opponents, int[::1] white_opp_totals, int[::1] black_opp_totals, int[::1] indices, double WA):
    cdef int j, k
    cdef double score, rating_white, rating_black, opp_white, opp_black

    for j in prange(errors.shape[0], nogil=True, schedule='guided'):
        score = 0.0
        rating_white = WA * ratings[j]
        rating_black = ratings[j]

        for k in range(indices[j], indices[j + 1]):
            opp_black = ratings[opponents[k]]
            opp_white = WA * ratings[opponents[k]]

            score = score + white_opp_totals[k] * rating_white / (rating_white + opp_black) \
                          + black_opp_totals[k] * rating_black / (rating_black + opp_white) 
        errors[j] = scored[j] - score


def adjust(double[::1] ratings, double[::1] errors, double[::1] total_inv, double K):
    cdef int i
    
    for i in prange(errors.shape[0], nogil=True, schedule='guided'):
        ratings[i] = ratings[i] * 10 ** (K * errors[i] * total_inv[i])
