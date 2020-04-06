from sympy import sec
import numpy as np
import os
from BUDDHA.kNN_two_way_interaction_effects_reproducing_paper_experiments import run


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

    num_trials = 10

    synthetic_data_X = []
    synthetic_data_y = []

    results = [[[], []] for _ in range(10)]

    # Generate synthetic data
    for i in range(10):
        print("\nExperiment ", i)
        for trial in range(num_trials):
            print("Trial \n", trial)
            for _ in range(size_of_synthetic_dataset):
                inputs = list(np.random.uniform(-1, 1, input_dimension))
                output = globals()["f_{}".format(i + 1)]()(inputs)

                synthetic_data_X.append(inputs)
                synthetic_data_y.append(output)

            # Output data and interaction effects
            data_directory_name = "synthetic_data_f_{}".format(i + 1)
            if not os.path.exists(root_data_directory + "/" + data_directory_name):
                os.makedirs(root_data_directory + "/" + data_directory_name)
            with open(root_data_directory + "/" + data_directory_name + "/X", "w") as file:
                file.write(str(synthetic_data_X))
            with open(root_data_directory + "/" + data_directory_name + "/y", "w") as file:
                file.write(str(synthetic_data_y))

            LR_score, kNN_score = run(i + 1)

            results[i][0].append(LR_score)
            results[i][1].append(kNN_score)

    trials_averaged = [[np.mean(t[0]), np.mean(t[1])] for t in results]
    print(results)
    print(trials_averaged)








