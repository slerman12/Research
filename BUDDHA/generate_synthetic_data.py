from sympy import sec
import numpy as np
from itertools import combinations
import os


# Given weights and values, compute the weighted sum of those values and their interactions
# Weights is a list of lists (weights) sorted by the order of the interaction
# If one of the orders of weights is "random", then random weights are generated for that order
# If weights is a scalar, then random weights are generated up to the order of that scalar
# If weights is None, then a weight of 1 is used and the order depends on the dimension of interactions
# Interactions specifies the indices included in the weighing and summing, as a list of lists sorted by order
# If one of the orders of interactions is None, then all interactions of that order are used
# If interactions is None, then all interactions up to the order of weights are used
# Default interaction is multiplicative
# e.g. values = [8, 2], weights = [[3, 4], [5]] --> 3(8) + 4(2) + 5(8*2) = 112
# The different "to do" markers indicate ways that this function can be generalized to arbitrary entity interactions
def weighted_sum_of_interactions(values, weights=3, interactions=None, interaction_func=lambda x: np.prod(x), noise=0,
                                 debug_mode=False):
    # Make sure values is a list/array
    assert type(values).__module__ == np.__name__ or isinstance(values, list)

    # Values dimension
    dim = len(values)
    if debug_mode:
        print("dimension: ", dim)  # Debug code

    # Make sure either weights or interactions is provided
    assert not (weights is None and interactions is None)

    # Prepare scalar weights to be order for randomly generated weights
    if isinstance(weights, int):
        weights = ["random" for _ in range(weights)]

    # Interaction order
    orders = len(interactions) if weights is None else len(weights)
    if debug_mode:
        print("order ", orders)  # Debug code

    # Make sure order does not exceed number of values
    assert orders <= dim

    # Prepare None interactions to be all interactions up to order
    if interactions is None:
        interactions = [None for _ in range(orders)]

    # Make sure interactions includes a dimension for each order
    assert len(interactions) == orders

    # Initialize the weighted sum  TODO adaptive shape
    weighted_sum = 0

    # Weights used
    weights_used = None if weights is None else [[] for _ in range(orders)]

    # Interactions applied
    interactions_applied = [[] for _ in range(orders)]

    # Sum interactions for each order
    for order in range(orders):
        # Make sure weights are a list/array or "random"
        if weights is not None:
            assert type(weights[order]).__module__ == np.__name__ or isinstance(weights[order], list) or weights[order] == "random"

        # Interactions for this order
        if interactions[order] is None:
            # TODO Could add cache per order, could use indices, could sort indices,
            #  could store hash in dict for fast viability check, and could add use a constrained version
            interactions[order] = list(combinations(values, order + 1))
        else:
            if debug_mode:
                print("interactions' indices for order ", order, ": ", interactions[order])  # Debug code
            # TODO use a separate interactions list internal to this function or else the one passed in gets overwritten
            interactions[order] = [[values[ind] for ind in interaction] for interaction in interactions[order]]
        if debug_mode:
            print("interactions for order ", order, ": ", interactions[order])  # Debug code

        # Assert that each dimension matches the dimension of unique value interactions for the respective order
        if weights[order] != "random":
            if debug_mode:
                print("dimension of interactions: ", len(interactions[order]))  # Debug code
                print("dimension of weights: ", len(weights[order]), "\n")  # Debug code
            assert len(weights[order]) == len(interactions[order])

        # Compute weighted interactions for order
        for i, interaction in enumerate(interactions[order]):
            # TODO Could allow weights to be a function of the interaction
            weight = 1 if weights is None else np.random.rand() if weights[order] == "random" or weights[order][i] == "random" else weights[order][i]
            interaction_applied = interaction_func(interaction)
            interactions_applied[order].append(interaction_applied)
            scaled_by_weight = weight * interaction_applied
            weighted_sum += scaled_by_weight
            if weights is not None:
                weights_used[order].append(weight)
            if debug_mode:
                print("weight: ", weight)  # Debug code
                print("interaction: ", interaction)  # Debug code
                print("applying interaction function: ", interaction_applied)  # Debug code
                print("scaling by weight: ", scaled_by_weight, "\n")  # Debug code

    # Return weighted sum
    if debug_mode:
        print("weighted sum: ", weighted_sum)  # Debug code
        print("weights used: ", weights_used)  # Debug code
    return weighted_sum, weights_used, interactions_applied


def f_1(x):
    return np.math.pi ** (x[0] * x[1]) * np.math.sqrt(2 * x[2]) - (1/np.math.sin(x[3])) + np.math.log(x[2] + x[4]) - \
           (x[8] / x[9]) * np.math.sqrt(x[6] / x[7]) - x[1] * x[6]


def f_2(x):
    return np.pi ** (x[0] * x[1]) * np.sqrt(2 * np.abs(x[2])) - (np.arcsin(0.5 * x[3])) + np.log(np.abs(x[2] + x[4]) + 1) - \
           (x[8] / (1 + np.abs(x[9]))) * np.sqrt(x[6] / (1 + np.abs(x[7]))) - x[1] * x[6]


def f_3(x):
    return np.exp(np.abs(x[0] - x[1])) + np.abs(x[1] * x[2]) - ((x[2]) ** 2) ** np.absolute(x[3]) + \
           np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + x[8] + 1 / (1 + x[9] ** 2)


def f_4(x):
    return np.exp(np.abs(x[0] - x[1])) + np.abs(x[1] * x[2]) - ((x[2]) ** 2) ** np.absolute(x[3]) + (x[0] * x[3]) ** 2 \
           + np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + x[8] + 1 / (1 + x[9] ** 2)


def f_5(x):
    return 1 / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + np.sqrt(np.exp(x[3] + x[4])) + np.abs(x[5] + x[6]) + x[7] * x[8] * x[9]


def f_6(x):
    return np.exp(np.abs(x[0] * x[1]) + 1) - np.exp(np.abs(x[2] + x[3]) + 1) + np.cos(x[4] + x[5] - x[7]) + np.sqrt(x[7] ** 2 + x[8] ** 2 + x[9] ** 2)


def f_7(x):
    return (np.arctan(x[0]) + np.arctan(x[1])) ** 2 + np.max(x[2] * x[3] + x[5], 0) - 1 / (1 + (x[3] * x[4] * x[5] * x[6] * x[7]) ** 2) + (np.abs(x[6]) / (1 + np.abs(x[8]))) ** 5 + np.sum([x[i] for i in range(10)])


def f_8(x):
    return x[0] * x[1] + 2 ** (x[2] + x[4] + x[5]) + 2 ** (x[2] + x[3] + x[4] + x[6]) + np.sin(x[6] * np.sin(x[7] + x[8])) + np.arccos(0.0 * x[9])


def f_9(x):
    return np.tanh(x[0] * x[1] + x[2] * x[3]) * np.sqrt(np.abs(x[4])) + np.exp(x[4] + x[5]) + np.log((x[5] * x[6] * x[7]) ** 2 + 1) + x[8] * x[9] + 1 / (1 + np.abs(x[9]))


def f_10(x):
    return np.sinh(x[0] + x[1]) + np.arccos(np.tanh(x[2] + x[4] + x[6])) + np.cos(x[3] + x[4]) + sec(x[6] * x[8])


if __name__ == "__main__":
    root_data_directory = "data/synthetic_data"
    size_of_synthetic_dataset = 30000
    input_dimension = 10
    # num_orders = 2

    synthetic_data_X = []
    synthetic_data_y = []

    # Randomly sample interaction effects
    # _, interaction_effects, _ = weighted_sum_of_interactions(np.zeros(input_dimension), num_orders)

    # Generate synthetic data
    for _ in range(size_of_synthetic_dataset):
        inputs = list(np.random.uniform(-1, 1, input_dimension))
        # output, _, _ = weighted_sum_of_interactions(inputs, interaction_effects)
        output = f_3(inputs)

        synthetic_data_X.append(inputs)
        synthetic_data_y.append(output)

    # Output data and interaction effects
    # data_directory_name = "synthetic_data_size_{}_input_dimension_{}_num_orders_{}_multiplicative_interactions_noise_0".format(size_of_synthetic_dataset, input_dimension, num_orders)
    data_directory_name = "synthetic_data_f_3"
    if not os.path.exists(root_data_directory + "/" + data_directory_name):
        os.makedirs(root_data_directory + "/" + data_directory_name)
    with open(root_data_directory + "/" + data_directory_name + "/X", "w") as file:
        file.write(str(synthetic_data_X))
    with open(root_data_directory + "/" + data_directory_name + "/y", "w") as file:
        file.write(str(synthetic_data_y))
    # with open(root_data_directory + "/" + data_directory_name + "/interaction_effects", "w") as file:
    #     file.write(str(interaction_effects))






