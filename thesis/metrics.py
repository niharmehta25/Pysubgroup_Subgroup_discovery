import numpy as np


def arl(y, y_prob):
    sorted_indices = np.argsort(y)
    sorted_true = y[sorted_indices]
    sorted_ranking = y_prob[sorted_indices]

    true_indices = np.where(sorted_true == 1)[0]
    false_indices = np.where(sorted_true == 0)[0]

    false_rankings = sorted_ranking[false_indices]
    higher_rankings = false_rankings > np.expand_dims(sorted_ranking[true_indices], axis=1)
    equal_rankings = false_rankings == np.expand_dims(sorted_ranking[true_indices], axis=1)

    PENNi_sum = np.sum(higher_rankings, axis=1) + 0.5 * np.sum(equal_rankings, axis=1)
    numerator_sum = np.sum(sorted_true[true_indices] * PENNi_sum)
    denominator_sum = np.sum(y)

    return numerator_sum / denominator_sum
