#!/usr/bin/env python
import json
import math
import operator
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression


class KNNInteractionEffects(object):
    def __init__(self, _X, _y, k):
        assert (k <= len(_X)), "k cannot be greater than training_set length"
        self.__X = _X.copy()
        self.__y = _y.copy()
        self.__k = k

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def gaussian(dist, sigma=1):
        return 1. / (math.sqrt(2. * math.pi) * sigma) * math.exp(-dist ** 2 / (2 * sigma ** 2))

    def compute_interaction_effects(self, x, _order=2, _interactions=None):
        feature_indices = list(range(len(self.__X[0])))
        _interactions = list(combinations(feature_indices, 2)) if _interactions is None else _interactions
        _interactions_length = len(_interactions)

        upper_lowers = np.zeros((_interactions_length, 2, 2))
        total_upper_lowers_weights = np.zeros((_interactions_length, 2, 2))

        distances = []
        for idx, d in enumerate(self.__X):
            dist = self.euclidean_distance(x, d)
            distances.append((self.__y[idx], dist, d - x))
        distances.sort(key=operator.itemgetter(1))

        for i in range(self.__k):
            weight = self.gaussian(distances[i][1])

            for j, _interaction in enumerate(_interactions):
                if distances[i][2][_interaction[0]] < 0:
                    if distances[i][2][_interaction[1]] < 0:
                        upper_lowers[j, 0, 0] += distances[i][0] * weight
                        total_upper_lowers_weights[j, 0, 0] += weight
                    else:
                        upper_lowers[j, 0, 1] += distances[i][0] * weight
                        total_upper_lowers_weights[j, 0, 1] += weight
                else:
                    if distances[i][2][_interaction[1]] < 0:
                        upper_lowers[j, 1, 0] += distances[i][0] * weight
                        total_upper_lowers_weights[j, 1, 0] += weight
                    else:
                        upper_lowers[j, 1, 1] += distances[i][0] * weight
                        total_upper_lowers_weights[j, 1, 1] += weight
        total_upper_lowers_weights[total_upper_lowers_weights == 0] = 1
        upper_lowers /= total_upper_lowers_weights

        return np.abs((upper_lowers[:, 0, 0] - upper_lowers[:, 1, 0]) - (upper_lowers[:, 0, 1] - upper_lowers[:, 1, 1]))


def _linear_regression_two_way_interaction_effects(_x, _y):
    _feature_indices = list(range(len(_x[0])))
    _x_plus_interaction_terms = _x.copy()
    for _interaction in combinations(_feature_indices, 2):
        _x_plus_interaction_terms = np.c_[_x_plus_interaction_terms, _x_plus_interaction_terms[:, _interaction[0]] *
                                          _x_plus_interaction_terms[:, _interaction[1]]]
    _model = LinearRegression(fit_intercept=False).fit(_x_plus_interaction_terms, _y)
    return _model.coef_[len(_x[0]):]


def rank_sequentially(inputs):
    output = [0] * len(inputs)
    for i, x in enumerate(sorted(range(len(inputs)), key=lambda y: inputs[y])):
        output[x] = i
    return output


def scale_to_0_1(l):
    return (np.array(l) - np.array(l).min(0)) / np.array(l).ptp(0)


def mae(_pred, _true):
    return np.mean(np.abs(np.array(_pred) - np.array(_true)))


def print_results(pred, true):
    print(pred)
    print(true)
    print()
    predicted = scale_to_0_1(pred)
    ground_truth = scale_to_0_1(true)
    print("predicted IE: ", predicted)
    print("ground truth: ", ground_truth)
    print("MAE: ", mae(predicted, ground_truth))
    predicted_ranked_sequentially = rank_sequentially(pred)
    ground_truth_ranked_sequentially = rank_sequentially(true)
    print("predicted IE ranked: ", predicted_ranked_sequentially)
    print("ground truth ranked: ", ground_truth_ranked_sequentially)
    print("ranked MAE: ", mae(predicted_ranked_sequentially, ground_truth_ranked_sequentially))
    print()


# Hyper-parameters
order = 2
samples = 100

# Load data
data_path = "data/synthetic_data/synthetic_data_size_5000_input_dimension_5_num_orders_2_multiplicative_interactions_noise_0"
with open(data_path + "/X") as f:
    X = json.load(f)
with open(data_path + "/y") as f:
    y = json.load(f)
with open(data_path + "/interaction_effects") as f:
    interaction_effects = json.load(f)

# Data normalization
X = np.array(X)
X = (X - X.min(0)) / X.ptp(0)

# Linear regression two way interaction effects
lin_reg_preds = _linear_regression_two_way_interaction_effects(X, y)
print_results(lin_reg_preds, interaction_effects[1])

# Two way interaction effects
two_way_interaction_effects = []
for sample in range(samples):
    model = KNNInteractionEffects(np.roll(X, sample, 0)[1:], np.roll(y, sample, 0)[1:], 4999)
    two_way_interaction_effects.append(model.compute_interaction_effects(X[-sample]))
two_way_interaction_effects = np.mean(two_way_interaction_effects, 0)
print_results(two_way_interaction_effects, interaction_effects[1])
