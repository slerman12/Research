#!/usr/bin/env python
import json
import math
import operator
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression


class kNN_interaction_effects(object):
    def __init__(self, x, y, k, weighted=False):
        assert (k <= len(x)), "k cannot be greater than training_set length"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def gaussian(dist, sigma=1):
        return 1. / (math.sqrt(2. * math.pi) * sigma) * math.exp(-dist ** 2/(2 * sigma ** 2))

    def compute_interaction_effects(self, test_set, interaction=None):
        predictions = []
        for t in test_set:
            distances = []
            for idx, d in enumerate(self.__x):
                dist = self.euclidean_distance(t, d)
                distances.append((self.__y[idx], dist, t[interaction[0]] > d[interaction[0]], t[interaction[1]] > d[interaction[1]],
                                  d[interaction[0]] - t[interaction[0]], d[interaction[1]] - t[interaction[1]]))
            distances.sort(key=operator.itemgetter(1))
            upper_0_upper_1 = 0
            upper_0_lower_1 = 0
            lower_0_upper_1 = 0
            lower_0_lower_1 = 0
            total_upper_0_upper_1_weight = 0
            total_upper_0_lower_1_weight = 0
            total_lower_0_upper_1_weight = 0
            total_lower_0_lower_1_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                # weight = 1.0 / (distances[i][1] + 0.1)
                if distances[i][2]:
                    if distances[i][3]:
                        upper_0_upper_1 += distances[i][0]*weight
                        total_upper_0_upper_1_weight += weight
                    else:
                        upper_0_lower_1 += distances[i][0]*weight
                        total_upper_0_lower_1_weight += weight
                else:
                    if distances[i][3]:
                        lower_0_upper_1 += distances[i][0]*weight
                        total_lower_0_upper_1_weight += weight
                    else:
                        lower_0_lower_1 += distances[i][0]*weight
                        total_lower_0_lower_1_weight += weight
            total_upper_0_upper_1_weight = 1 if total_upper_0_upper_1_weight == 0 else total_upper_0_upper_1_weight
            total_upper_0_lower_1_weight = 1 if total_upper_0_lower_1_weight == 0 else total_upper_0_lower_1_weight
            total_lower_0_upper_1_weight = 1 if total_lower_0_upper_1_weight == 0 else total_lower_0_upper_1_weight
            total_lower_0_lower_1_weight = 1 if total_lower_0_lower_1_weight == 0 else total_lower_0_lower_1_weight
            preds = [[upper_0_upper_1 / total_upper_0_upper_1_weight, upper_0_lower_1 / total_upper_0_lower_1_weight],
                     [lower_0_upper_1 / total_lower_0_upper_1_weight, lower_0_lower_1 / total_lower_0_lower_1_weight]]
            predictions.append(abs((preds[0][0] - preds[1][0]) - (preds[0][1] - preds[1][1])))
        return predictions


def _linear_regression_two_way_interaction_effects(_x, _y):
    _feature_indices = list(range(len(_x[0])))
    _x_plus_interaction_terms = _x.copy()
    for _interaction in combinations(_feature_indices, 2):
        _x_plus_interaction_terms = np.c_[_x_plus_interaction_terms, _x_plus_interaction_terms[:, _interaction[0]] *
                                          _x_plus_interaction_terms[:, _interaction[1]]]
    _model = LinearRegression(fit_intercept=False).fit(_x_plus_interaction_terms, _y)
    return _model.coef_[len(_x[0]):]


def compute_two_way_interaction_effects(_X, _y, _samples):
    _X = _X.copy()
    _y = _y.copy()
    feature_indices = list(range(len(_X[0])))
    result = []
    for sample in range(_samples):
        two_way_interaction_effects_running = []

        for two_way_interaction in combinations(feature_indices, 2):
            model = kNN_interaction_effects(np.array(_X[1:]), _y[1:], 4999, True)
            predicted_interaction_effect = model.compute_interaction_effects(np.array([_X[0]]),
                                                                             interaction=two_way_interaction)[0]
            two_way_interaction_effects_running.append(predicted_interaction_effect)

        if len(result) == 0:
            result = np.array(two_way_interaction_effects_running)
        else:
            result += np.array(two_way_interaction_effects_running)
        _X = np.roll(_X, 1, 0)
        _y = np.roll(_y, 1, 0)
    return result / samples


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

# Two way interaction effects
two_way_interaction_effects = compute_two_way_interaction_effects(X, y, samples)
print_results(two_way_interaction_effects, interaction_effects[1])

# Linear regression two way interaction effects
lin_reg_preds = _linear_regression_two_way_interaction_effects(X, y)
print_results(lin_reg_preds, interaction_effects[1])
