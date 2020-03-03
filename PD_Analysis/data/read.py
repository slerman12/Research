import pandas as pd
import numpy as np
import random


# Data handler
class ReadPD:
    def __init__(self, filename, targets, train_test_split=1, valid_eval_split=0, cross_val_folds=None,
                 inference_type="future_scores_one_to_one", to_drop=None, features=None):
        # Processed data file
        data = pd.read_csv(filename)
        self.data = data

        # Set variables
        self.targets = targets
        self.train_test_split = train_test_split
        self.valid_eval_split = valid_eval_split
        self.cross_val_folds = cross_val_folds
        self.inference_type = inference_type

        # Set features (with defaults)
        self.to_drop = ["PATNO", "INFODT"] if to_drop is None else ["PATNO", "INFODT"] + to_drop
        self.features = list(data.columns.values) if features is None else features
        self.features = [x for x in self.features if x not in self.to_drop + self.targets]
        print("\nDropped variables:")
        print(self.to_drop)
        print("\nSelected variables:")
        print(self.features)
        print("\nTarget summary:")
        print(data[self.targets].describe())

        # Variables for iterating batches
        self.batch_begin = 0
        self.epoch_complete = False
        self.make_batch = lambda x: x

        # Patients
        patients = list(data["PATNO"].unique())

        # Shuffle patients TODO: sample an int randomly to use as seed for all runs, same for batch iteration
        random.Random().shuffle(patients)

        # Splits data according to patients TODO validation should be independent of the train test split for cross val
        def split(train_test, valid_eval, offset=0):
            # TODO: offset (for folds) - can perhaps just reorder patients temporarily

            # Split patients
            training_data_patients = list(patients[:round(train_test * len(patients))])
            testing_data_patients = patients[round(train_test * len(patients)):]
            valid_eval_split_index = round(valid_eval * len(testing_data_patients))
            validation_data_patients = testing_data_patients[:valid_eval_split_index]
            evaluation_data_patients = testing_data_patients[valid_eval_split_index:]

            # Split data
            training = data[data["PATNO"].isin(training_data_patients)]
            validation = data[data["PATNO"].isin(validation_data_patients)]
            evaluation = data[data["PATNO"].isin(evaluation_data_patients)]

            # Return split data
            return training, validation, evaluation

        # Either set data directly if cross validation folds is None, or do multiple folded splits otherwise
        if cross_val_folds is None:
            # Split data
            training_data, validation_data, evaluation_data = split(train_test_split, valid_eval_split)
            self.training = training_data

            # Create data
            self.training_data_list = self.start(training_data)
            self.validation_data = self.make_batch(self.start(validation_data))
            self.evaluation_data = self.make_batch(self.start(evaluation_data))

            # Training data size
            self.training_data_size = len(self.training_data_list)

            print("Training data size: ", self.training_data_size)
            print("Validation data size: ", self.validation_data[0].shape[0])
            print("Testing data size: ", self.evaluation_data[0].shape[0])
        else:
            # Initialize folds
            self.training_data_list_folds = []
            self.evaluation_data_folds = []

            # For each fold
            for fold in range(self.cross_val_folds):
                # Create data for folds
                training_data, _, evaluation_data = split(train_test_split, 0, fold)
                self.training_data_list_folds.append(self.start(training_data))
                self.evaluation_data_folds.append(self.make_batch(self.start(evaluation_data)))

            # Initialize to first fold
            self.set_fold(0)

        # All data
        self.all_data = self.make_batch(self.start(data))

    def start(self, data):
        # Initiate data array
        data_list = []

        # Initialize make_batch function
        make_batch = None

        # Initiate data according to inference type
        if self.inference_type == "future_scores_one_to_one":
            # Iterate through all rows and append data as dictionary to data_list
            data_list = data.to_dict('records')
            # for index, observation in data.iterrows():
            #     data_list.append({"id": observation.at["PATNO"],
            #                       "inputs": observation[self.features].drop(self.targets).values,
            #                       "desired_outputs": observation[self.targets].values,
            #                       "time_ahead": observation["TIME_AHEAD"].item()})

            # Define make batch function depending on inference type
            def make_batch(d):
                return np.stack([[element[feature] for feature in self.features] for element in d]), \
                       np.stack([[element[feature] for feature in self.targets] for element in d]), \
                       np.stack([element["TIME_AHEAD"] for element in d])
                # return np.stack([element["inputs"] for element in d]), \
                #        np.stack([element["desired_outputs"] for element in d]), \
                #        np.stack([element["time_ahead"] for element in d])

        # Initiate data according to inference type
        if self.inference_type == "rates_one_to_one":
            # Iterate through all rows and append data as dictionary to data_list
            data_list = data.to_dict('records')

            # Define make batch function depending on inference type
            def make_batch(d):
                return np.stack([[element[feature] for feature in self.features] for element in d]), \
                       np.stack([[element[feature] for feature in self.targets] for element in d])

        # Set make_batch function
        self.make_batch = make_batch

        # Shuffle data list
        random.shuffle(data_list)

        # Return data list
        return data_list

    def iterate_batch(self, batch_size):  # TODO perhaps balance classes for classification
        # Reset and shuffle batch when all items have been iterated
        if self.batch_begin > self.training_data_size - batch_size:
            # Reset batch index
            self.batch_begin = 0

            # Shuffle PD data
            random.shuffle(self.training_data_list)

        # Index of the end boundary of this batch
        batch_end = min(self.batch_begin + batch_size, self.training_data_size)

        # Batch
        batch = self.make_batch(self.training_data_list[self.batch_begin:batch_end])

        # Update batch index
        self.batch_begin = batch_end

        # Update epoch
        self.epoch_complete = self.batch_begin > self.training_data_size - batch_size

        # Return batch
        return batch

    def set_fold(self, fold):
        # Set data
        self.training_data_list = self.training_data_list_folds[fold]
        self.evaluation_data = self.evaluation_data_folds[fold]

        # Set training data size
        self.training_data_size = len(self.training_data_list)

        # Reset batch iteration
        self.batch_begin = 0
        self.epoch_complete = False

    def class_balancing(self):
        # Oversampling
        max_size = self.training[self.targets[0]].value_counts().max()

        lst = [self.training]
        for class_index, group in self.training.groupby(self.targets[0]):
            lst.append(group.sample(max_size-len(group), replace=True))
        self.balanced_training = pd.concat(lst)

        self.training_data_list = self.start(self.balanced_training)

        # Training data size
        self.training_data_size = len(self.training_data_list)
