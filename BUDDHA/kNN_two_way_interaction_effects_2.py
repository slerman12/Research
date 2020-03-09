#!/usr/bin/env python
import json
import math
import operator
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

    def predict(self, test_set, dx):
        predictions = []
        for t in test_set:
            distances = []
            for idx, d in enumerate(self.__x):
                dist = self.euclidean_distance(t, d)
                distances.append((self.__y[idx], dist, t[dx] > d[dx], d[dx] - t[dx]))
            distances.sort(key=operator.itemgetter(1))
            upper_v = 0
            lower_v = 0
            total_upper_weight = 0
            total_lower_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                # weight = 1.0 / (distances[i][1] + 0.001)
                if distances[i][2]:
                    if self.__weighted:
                        upper_v += distances[i][0]*weight
                    else:
                        upper_v += distances[i][0]
                    total_upper_weight += weight
                else:
                    if self.__weighted:
                        lower_v += distances[i][0]*weight
                    else:
                        lower_v += distances[i][0]
                    total_lower_weight += weight
            if self.__weighted:
                predictions.append(((upper_v/total_upper_weight) - (lower_v/total_lower_weight)))
                # predictions.append(LinearRegression(False).fit(np.expand_dims(np.array(distances)[:, 3], axis=1), np.array(distances)[:, 0]).coef_[0])
            else:
                predictions.append((upper_v/self.__k) - (lower_v/self.__k))
        return predictions


order = 2

with open("data/synthetic_data/synthetic_data_size_5000_input_dimension_5_num_orders_2_multiplicative_interactions_noise_0/X") as f:
    X = json.load(f)
with open("data/synthetic_data/synthetic_data_size_5000_input_dimension_5_num_orders_2_multiplicative_interactions_noise_0/y") as f:
    y = json.load(f)
with open("data/synthetic_data/synthetic_data_size_5000_input_dimension_5_num_orders_2_multiplicative_interactions_noise_0/interaction_effects") as f:
    interaction_effects = json.load(f)

X = np.array(X)
X = (X - X.min(0)) / X.ptp(0)


def compute_two_way_interaction_effect(_interaction):
    full_model = kNN(np.array(X[1:]), y[1:], 4999, True)

    # Compute dx for 0th component of interaction on full inputs
    dx_0_full = full_model.predict(np.array([X[0]]), (_interaction[0],))[0]

    test_inputs_sans_1 = np.array([X[0]])
    test_inputs_sans_1[:, _interaction[1]] = 0
    inputs_sans_1 = np.array(X[1:])
    inputs_sans_1[:, _interaction[1]] = 0
    model = kNN(inputs_sans_1, y[1:], 4999, True)
    dx_0_partial = model.predict(test_inputs_sans_1, (_interaction[0],))[0]

    dx_1_full = full_model.predict(np.array([X[0]]), (_interaction[1],))[0]

    test_inputs_sans_0 = np.array([X[0]])
    test_inputs_sans_0[:, _interaction[0]] = 0
    inputs_sans_0 = np.array(X[1:])
    inputs_sans_0[:, _interaction[0]] = 0
    model = kNN(inputs_sans_0, y[1:], 4999, True)
    dx_1_partial = model.predict(test_inputs_sans_0, (_interaction[1],))[0]

    # return dx_0_full - dx_0_partial + dx_1_full - dx_1_partial
    # return (dx_0_full - dx_0_partial)**2 + (dx_1_full - dx_1_partial)**2
    return max((dx_0_full - dx_0_partial), (dx_1_full - dx_1_partial))
    # return max(abs(dx_0_full - dx_0_partial), abs(dx_1_full - dx_1_partial))
    # return max((dx_0_full - dx_0_partial)**2, (dx_1_full - dx_1_partial)**2)


samples = 100

feature_indices = list(range(len(X[0])))
two_way_interaction_effects = []
for sample in range(samples):
    two_way_interaction_effects_running = []

    for interaction in combinations(feature_indices, 2):
        two_way_interaction_effects_running.append(compute_two_way_interaction_effect(interaction))

    if len(two_way_interaction_effects) == 0:
        two_way_interaction_effects = np.array(two_way_interaction_effects_running)
    else:
        two_way_interaction_effects += np.array(two_way_interaction_effects_running)

    X = np.roll(X, 1, 0)
    y = np.roll(y, 1, 0)


def rank(inputs):
    output = [0] * len(inputs)
    for i, x in enumerate(sorted(range(len(inputs)), key=lambda y: inputs[y])):
        output[x] = i

    return output


print(two_way_interaction_effects)
print(interaction_effects[1])
print()
print((np.array(two_way_interaction_effects) - np.array(two_way_interaction_effects).min(0)) / np.array(two_way_interaction_effects).ptp(0))
print((np.array(interaction_effects[1]) - np.array(interaction_effects[1]).min(0)) / np.array(interaction_effects[1]).ptp(0))
print()
print("predicted IE: ", rank(two_way_interaction_effects))
print("ground truth: ", rank(interaction_effects[1]))
print(np.linalg.norm(np.array(rank(two_way_interaction_effects)) - np.array(rank(interaction_effects[1]))))
print()


# print(np.argsort(predicted_interaction_effects[1]) ** 2 - np.argsort(interaction_effects[1]) ** 2)
# print(np.mean(np.sqrt(np.argsort(predicted_interaction_effects[1]) ** 2 - np.argsort(interaction_effects[1]) ** 2)))

def _linear_regression_two_way_interaction_effects(_x, _y):
    _feature_indices = list(range(len(_x[0])))
    _x_plus_interaction_terms = _x.copy()
    for _interaction in combinations(_feature_indices, 2):
        _x_plus_interaction_terms = np.c_[_x_plus_interaction_terms, _x_plus_interaction_terms[:, _interaction[0]] *
                                          _x_plus_interaction_terms[:, _interaction[1]]]
    _model = LinearRegression(fit_intercept=False).fit(_x_plus_interaction_terms, _y)
    return _model.coef_[len(_x[0]):]


lin_reg_preds = _linear_regression_two_way_interaction_effects(X, y)
print()
print("lin reg pred: ", lin_reg_preds)
print(lin_reg_preds)
print(interaction_effects[1])
print()
print((np.array(lin_reg_preds) - np.array(lin_reg_preds).min(0)) / np.array(lin_reg_preds).ptp(0))
print((np.array(interaction_effects[1]) - np.array(interaction_effects[1]).min(0)) / np.array(interaction_effects[1]).ptp(0))
print()
print("predicted IE: ", rank(lin_reg_preds))
print("ground truth: ", rank(interaction_effects[1]))
print(np.linalg.norm(np.array(rank(lin_reg_preds)) - np.array(rank(interaction_effects[1]))))
