from __future__ import division

import itertools
import numpy as np

from iou import cc_iou as iou


def score_craters_on_patch(y_true, y_pred):
    """
    Main score

    Parameters
    ----------
    y_true : list of tuples
        List of coordinates and radius of craters in a patch (x, y, radius)
    y_pred : list of tuples
        List of coordinates and radius of craters found in the patch

    Returns
    -------
    float : score for a given path

    """
    # currently hard-coded
    p_norm = 1
    cut_off = 1

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ospa_score = ospa(y_true, y_pred, p_norm=p_norm, cut_off=cut_off)

    score = 1 - ospa_score

    return score


def score_iou(y_true, y_pred):
    pass


def score_completeness(y_true, y_pred):
    pass


def ospa(x_arr, y_arr, p_norm=1, cut_off=1):
    """
    Optimal Subpattern Assignment (OSPA) metric

    This metric provides a coherent way to compute the miss-distance
    between

    Parameters
    ----------
    x_arr :

    y_arr : list of tuples

    p_norm : int

    cut_off : float

    Returns
    -------

    References
    ----------
    http://www.dominic.schuhmacher.name/papers/ospa.pdf

    """
    _, m = x_arr.shape
    _, n = y_arr.shape

    if m > n:
        return ospa(y_arr, x_arr, p_norm, cut_off)

    # ARBITRARY THRESHOLD TO SAVE COMPUTING TIME
    if n > 4 * m and n > 15:
        return 1

    if m == 0:
        # GOOD MATCH OF NO CRATERS
        if n == 0:
            return 0
        # BAD MATCH OF NO CRATERS => cardinality penalty only
        return cut_off

    iou_score = 0
    permutation_indices = itertools.permutations(range(n), m)
    for idx in permutation_indices:
        new_dist = sum(iou(x_arr[:, j], y_arr[:, idx[j]]) ** p_norm
                       for j in range(m))
        iou_score = max(iou_score, new_dist)

    distance_score = m - iou_score
    cardinality_score = cut_off ** p_norm * (n - m)

    dist = (1 / n * (distance_score + cardinality_score)) ** (1 / p_norm)

    return dist
