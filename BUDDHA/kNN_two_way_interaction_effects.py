#!/usr/bin/env python
import json
import math
import operator
from itertools import combinations
import numpy as np


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

    def predict(self, test_set):
        predictions = []
        for t in test_set:
            distances = []
            for idx, d in enumerate(self.__x):
                dist = self.euclidean_distance(t, d)
                distances.append((self.__y[idx], dist))
            distances.sort(key=operator.itemgetter(1))
            v = 0
            total_weight = 0
            for i in range(self.__k):
                weight = self.gaussian(distances[i][1])
                weight = 1.0 / (distances[i][1] + 0.00001)
                if self.__weighted:
                    v += distances[i][0]*weight
                else:
                    v += distances[i][0]
                total_weight += weight
            if self.__weighted:
                predictions.append(v/total_weight)
            else:
                predictions.append(v/self.__k)
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


def compute_predictiveness(_interaction):
    test_interaction = np.array([[X[0][i] for i in _interaction]])
    inputs = np.array(X[1:])[:, _interaction]

    model = kNN(inputs, y[1:], 4999, True)
    pred = model.predict(test_interaction)

    return 1.0 / model.euclidean_distance(y[0], pred[0])


feature_indices = np.array(list(range(len(X[0]))))
samples = 1000

for sample in range(samples):
    predicted_interaction_effects = [None for _ in range(order)]
    predicted_interaction_effects_running = [[] for _ in range(order)]

    cache = {}
    for o in range(order):
        for interaction in list(combinations(feature_indices, o + 1)):
            interaction_name = str(np.sort(list(interaction)))
            cache[interaction_name] = compute_predictiveness(interaction)

            max_predictiveness = -math.inf

            sub_interactions = [0]

            if len(interaction) > 1:
                sub_interactions = list(combinations(interaction, len(interaction) - 1))
                for sub_interaction in sub_interactions:
                    sub_interaction_name = str(np.sort(list(sub_interaction)))
                    if sub_interaction_name not in cache:
                        cache[sub_interaction_name] = compute_predictiveness(sub_interaction)

                    # print(cache[sub_interaction_name])
                    if cache[sub_interaction_name] > max_predictiveness:
                        max_predictiveness = cache[sub_interaction_name]
                    # max_predictiveness += cache[sub_interaction_name]

            else:
                max_predictiveness = np.mean(y[1:])

            # print(cache[interaction_name])
            # print()
            # interaction_effect = cache[interaction_name] - max_predictiveness / len(sub_interactions)
            interaction_effect = cache[interaction_name] - max_predictiveness
            predicted_interaction_effects_running[o].append(interaction_effect)

        if predicted_interaction_effects[o] is None:
            predicted_interaction_effects[o] = np.array(predicted_interaction_effects_running[o])
        else:
            predicted_interaction_effects[o] += np.array(predicted_interaction_effects_running[o])
    temp = X[-1]
    X[-1] = X[0]
    X[0] = temp
    temp = y[-1]
    y[-1] = y[0]
    y[0] = temp


def rank(inputs):
    output = [0] * len(inputs)
    for i, x in enumerate(sorted(range(len(inputs)), key=lambda y: inputs[y])):
        output[x] = i

    return output


print(predicted_interaction_effects_running[1])
print(interaction_effects[1])
print()
print((np.array(predicted_interaction_effects_running[1]) - np.array(predicted_interaction_effects_running[1]).min(0)) / np.array(predicted_interaction_effects_running[1]).ptp(0))
print((np.array(interaction_effects[1]) - np.array(interaction_effects[1]).min(0)) / np.array(interaction_effects[1]).ptp(0))
print()
print(rank(predicted_interaction_effects_running[1]))
print(rank(interaction_effects[1]))
print()
print(rank(predicted_interaction_effects_running[0]))
print(rank(interaction_effects[0]))


# print(np.argsort(predicted_interaction_effects[1]) ** 2 - np.argsort(interaction_effects[1]) ** 2)
# print(np.mean(np.sqrt(np.argsort(predicted_interaction_effects[1]) ** 2 - np.argsort(interaction_effects[1]) ** 2)))
