#!/usr/bin/env python
import json
import math
import operator
import random
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression


class kNN(object):
    def __init__(self, x, y, k, weighted=False):
        assert (k <= len(x)
                ), "k cannot be greater than training_set length"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a-b)

    @staticmethod
    def gaussian(dist, sigma=1):
        return 1./(math.sqrt(2.*math.pi)*sigma)*math.exp(-dist**2/(2*sigma**2))

    def predict(self, test_set, dxdy):
        predictions = []
        for t in test_set:
            distances = []
            for idx, d in enumerate(self.__x):
                dist = self.euclidean_distance(t, d)
                distances.append((self.__y[idx], dist, t[dxdy[0]] > d[dxdy[0]], t[dxdy[1]] > d[dxdy[1]],
                                  d[dxdy[0]] - t[dxdy[0]], d[dxdy[1]] - t[dxdy[1]]))
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


def compute_two_way_interaction_effect(_X, _y, _interaction):
    model = kNN(np.array(_X[1:]), _y[1:], 4999, True)
    return model.predict(np.array([_X[0]]), _interaction)[0]


def rank_sequentially(inputs):
    output = [0] * len(inputs)
    for i, x in enumerate(sorted(range(len(inputs)), key=lambda y: inputs[y])):
        output[x] = i
    return output


def scale_to_0_1(l):
    return (np.array(l) - np.array(l).min(0)) / np.array(l).ptp(0)


def print_results(pred, true):
    print(pred)
    print(true)
    print()
    predicted = scale_to_0_1(pred)
    ground_truth = scale_to_0_1(true)
    print("predicted IE: ", predicted)
    print("ground truth: ", ground_truth)
    print(np.linalg.norm(predicted - ground_truth))
    print("predicted IE ranked: ", rank_sequentially(pred))
    print("ground truth ranked: ", rank_sequentially(true))
    print(kNN.euclidean_distance(np.array(rank_sequentially(pred)), np.array(rank_sequentially(true))))
    print()


order = 2
samples = 100

data_path = "data/synthetic_data/synthetic_data_size_5000_input_dimension_5_num_orders_2_multiplicative_interactions_noise_0"
with open(data_path + "/X") as f:
    X = json.load(f)
with open(data_path + "/y") as f:
    y = json.load(f)
with open(data_path + "/interaction_effects") as f:
    interaction_effects = json.load(f)

X = np.array(X)
X = (X - X.min(0)) / X.ptp(0)

feature_indices = list(range(len(X[0])))
two_way_interaction_effects = []
for sample in range(samples):
    two_way_interaction_effects_running = []

    for interaction in combinations(feature_indices, 2):
        two_way_interaction_effects_running.append(compute_two_way_interaction_effect(X, y, interaction))

    if len(two_way_interaction_effects) == 0:
        two_way_interaction_effects = np.array(two_way_interaction_effects_running)
    else:
        two_way_interaction_effects += np.array(two_way_interaction_effects_running)
    X = np.roll(X, 1, 0)
    y = np.roll(y, 1, 0)
two_way_interaction_effects = two_way_interaction_effects / samples
print_results(two_way_interaction_effects, interaction_effects[1])

lin_reg_preds = _linear_regression_two_way_interaction_effects(X, y)
print_results(lin_reg_preds, interaction_effects[1])
