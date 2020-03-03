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


if __name__ == "__main__":
    root_data_directory = "data/synthetic_data"
    size_of_synthetic_dataset = 5000
    input_dimension = 5
    num_orders = 2

    synthetic_data_X = []
    synthetic_data_y = []

    # Randomly sample interaction effects
    _, interaction_effects, _ = weighted_sum_of_interactions(np.zeros(input_dimension), num_orders)

    # Generate synthetic data
    for _ in range(size_of_synthetic_dataset):
        inputs = list(np.random.rand(input_dimension))
        output, _, _ = weighted_sum_of_interactions(inputs, interaction_effects)

        synthetic_data_X.append(inputs)
        synthetic_data_y.append(output)

    # Output data and interaction effects
    data_directory_name = "synthetic_data_size_{}_input_dimension_{}_num_orders_{}_multiplicative_interactions_noise_0".format(size_of_synthetic_dataset, input_dimension, num_orders)
    if not os.path.exists(root_data_directory + "/" + data_directory_name):
        os.makedirs(root_data_directory + "/" + data_directory_name)
    with open(root_data_directory + "/" + data_directory_name + "/X", "w") as file:
        file.write(str(synthetic_data_X))
    with open(root_data_directory + "/" + data_directory_name + "/y", "w") as file:
        file.write(str(synthetic_data_y))
    with open(root_data_directory + "/" + data_directory_name + "/interaction_effects", "w") as file:
        file.write(str(interaction_effects))






