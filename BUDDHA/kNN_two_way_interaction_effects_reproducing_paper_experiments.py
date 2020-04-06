#!/usr/bin/env python
import json
import math
import time
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


class NearestNeighborsInteractionEffects(object):
    def __init__(self, _X, _y):
        self.__X = _X.copy()
        self.__y = _y.copy()

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def gaussian(dist, sigma=1):
        return 1. / (math.sqrt(2. * math.pi) * sigma) * math.exp(-dist ** 2 / (2 * sigma ** 2))

    def compute_interaction_effects(self, x, _order=2, _interactions=None):
        time_start = time.time()

        feature_indices = list(range(len(self.__X[0])))
        _interactions = list(combinations(feature_indices, _order)) if _interactions is None else _interactions
        _interactions_length = len(_interactions)

        upper_lowers = np.zeros((_interactions_length, _order, 2))
        total_upper_lowers_weights = np.zeros((_interactions_length, _order, 2))

        # TODO might be able to do with numpy without for loop
        for i, d in enumerate(self.__X):
            # TODO account for missing values
            dist = self.euclidean_distance(x, d)
            diff = d - x

            weight = self.gaussian(dist)

            for j, _interaction in enumerate(_interactions):
                assert len(_interaction) == _order
                update_indices = (j,) + tuple(1 if diff[dim] < 0 else 0 for dim in _interaction)
                upper_lowers[update_indices] += self.__y[i] * weight
                total_upper_lowers_weights[update_indices] += weight

        total_upper_lowers_weights[total_upper_lowers_weights == 0] = 1
        upper_lowers /= total_upper_lowers_weights

        # TODO this only works for two way interactions. Is there a dynamic programming way to do higher-order?
        _interaction_effects = np.abs((upper_lowers[:, 0, 0] - upper_lowers[:, 1, 0]) -
                                      (upper_lowers[:, 0, 1] - upper_lowers[:, 1, 1]))

        print("interaction effects computation time: ", time.time() - time_start)
        return _interaction_effects


def _linear_regression_two_way_interaction_effects(_X, _y):
    time_start = time.time()
    _feature_indices = list(range(len(_X[0])))
    _X_plus_interaction_terms = _X.copy()
    for _interaction in combinations(_feature_indices, 2):
        _X_plus_interaction_terms = np.c_[_X_plus_interaction_terms, _X_plus_interaction_terms[:, _interaction[0]] *
                                          _X_plus_interaction_terms[:, _interaction[1]]]
    _model = LinearRegression(fit_intercept=False).fit(_X_plus_interaction_terms, _y)
    print("linear regression computation time: ", time.time() - time_start)
    return _model.coef_[len(_X[0]):]


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


def AUC(pred, true, X):
    predicted = scale_to_0_1(pred)
    print("predicted IE: ", predicted)
    predicted_ranked_sequentially = rank_sequentially(pred)
    print("predicted IE ranked: ", predicted_ranked_sequentially)

    true = [sorted(i) for i in true]

    top_pred = np.argsort(pred)[-len(true):]

    predicted_interactions = []
    ground_truth_interactions = []

    _feature_indices = list(range(len(X[0])))
    for i, _interaction in enumerate(list(combinations(_feature_indices, 2))):
        if i in top_pred:
            predicted_interactions.append(1)
        else:
            predicted_interactions.append(0)
        if sorted(_interaction) in true:
            ground_truth_interactions.append(1)
        else:
            ground_truth_interactions.append(0)
        print("Interaction: {}, Predicted: {}, Ranking: {}".format(_interaction, predicted[i], predicted_ranked_sequentially[i]))

    score = roc_auc_score(ground_truth_interactions, predicted_interactions)
    print("AUC score: ", score)

    return score


def run(num):
    # Hyper-parameters
    order = 2

    # Load data
    data_path = "data/synthetic_data/synthetic_data_f_1"
    with open(data_path + "/X") as f:
        X = json.load(f)
    with open(data_path + "/y") as f:
        y = json.load(f)

    # Ground truths
    f_1_ground_truth = [(0, 1), (0, 2), (1, 2), (2, 4), (8, 9), (8, 6), (8, 7), (9, 6), (9, 7), (6, 7), (1, 6)]
    f_2_ground_truth = [(0, 1), (0, 2), (1, 2), (2, 4), (8, 9), (8, 6), (8, 7), (9, 6), (9, 7), (6, 7), (1, 6)]
    f_3_ground_truth = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 6), (3, 7), (4, 6), (4, 7), (6, 7)]
    f_4_ground_truth = []
    f_5_ground_truth = []
    f_6_ground_truth = []
    f_7_ground_truth = []
    f_8_ground_truth = []
    f_9_ground_truth = []
    f_10_ground_truth = []

    # Data normalization
    X = np.array(X)
    X = (X - X.min(0)) / X.ptp(0)

    # Linear regression two way interaction effects
    lin_reg_preds = _linear_regression_two_way_interaction_effects(X, y)
    LR_score = AUC(lin_reg_preds, locals()["f_{}_ground_truth".format(num + 1)], X)

    # Two way interaction effects
    two_way_interaction_effects = []

    # With sampling
    # samples = 100
    # for sample in range(samples):
    #     model = NearestNeighborsInteractionEffects(np.roll(X, sample, 0)[1:], np.roll(y, sample, 0)[1:])
    #     two_way_interaction_effects.append(model.compute_interaction_effects(X[-sample]))
    # two_way_interaction_effects = [np.mean(two_way_interaction_effects, 0)]
    # AUC(two_way_interaction_effects[0], f_1_ground_truth)

    # Efficient two way interaction effects from median & mean!
    # two_way_interaction_effects = []
    model = NearestNeighborsInteractionEffects(X, y)
    two_way_interaction_effects.append(model.compute_interaction_effects(np.median(X, 0)))
    two_way_interaction_effects.append(model.compute_interaction_effects(np.mean(X, 0)))
    two_way_interaction_effects = np.mean(two_way_interaction_effects, 0)
    kNN_score = AUC(two_way_interaction_effects, locals()["f_{}_ground_truth".format(num + 1)], X)

    return LR_score, kNN_score


# run(1)
