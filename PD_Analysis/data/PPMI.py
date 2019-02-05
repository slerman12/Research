import pandas as pd
import numpy as np
import scipy
from sklearn import preprocessing
import random


# Data handler
class ReadPD:
    def __init__(self, filename, targets, to_drop=None, train_test_split=1, train_memory_split=1, valid_eval_split=0,
                 sequence_dropout=False, temporal=True, inference_type="future_score", groups=None):
        # Processed data file
        file = pd.read_csv(filename)

        # Targets
        self.targets = targets

        self.groups = groups

        # Default drop list
        self.to_drop = ["PATNO", "INFODT"] if to_drop is None else ["PATNO", "INFODT"] + to_drop

        # Make sure time column is in date time format
        file["INFODT"] = pd.to_datetime(file["INFODT"])

        # Train memory split
        self.train_memory_split = train_memory_split

        # Train test split
        self.train_test_split = train_test_split

        # Testing split
        self.valid_eval_split = valid_eval_split

        # Sequence dropout
        self.sequence_dropout = sequence_dropout

        # Max number of records for any patient
        self.max_num_records = file.groupby(["PATNO"]).size().max() - 1

        # Dimension of a patient record
        self.patient_record_dim = len(file.drop(self.to_drop, axis=1).columns.values)

        # Dimension of input
        self.input_dim = self.patient_record_dim

        # Dimension of desired output
        self.desired_output_dim = len(targets)

        # Variable for iterating batches
        self.batch_begin = 0
        self.epoch_complete = False

        # Patients
        patients = list(file["PATNO"].unique())

        # Shuffle patients TODO: sample an int randomly to use as seed for all runs, same for batch iteration
        random.Random(10).shuffle(patients)

        # Split training sets
        self.training_memory_data_patients = list(patients[:round(self.train_test_split * len(patients))])
        self.train_memory_split_index = round(self.train_memory_split * len(self.training_memory_data_patients))
        self.training_data_patients = list(self.training_memory_data_patients[:self.train_memory_split_index])
        self.memory_data_patients = list(self.training_memory_data_patients[self.train_memory_split_index:])

        # Split testing sets
        self.testing_data_patients = patients[round(self.train_test_split * len(patients)):]
        self.valid_eval_split_index = round(self.valid_eval_split * len(self.testing_data_patients))
        self.validation_data_patients = self.testing_data_patients[:self.valid_eval_split_index]
        self.evaluation_data_patients = self.testing_data_patients[self.valid_eval_split_index:]

        # Assigns data sets
        self.training_memory_data_file = file[file["PATNO"].isin(self.training_memory_data_patients)]
        self.training_data_file = file[file["PATNO"].isin(self.training_data_patients)]
        self.memory_data_file = file[file["PATNO"].isin(self.memory_data_patients)]
        self.validation_data_file = file[file["PATNO"].isin(self.validation_data_patients)]
        self.evaluation_data_file = file[file["PATNO"].isin(self.evaluation_data_patients)]

        # Create data
        self.training_memory_data = self.start(self.training_memory_data_file, temporal=temporal, inference_type=inference_type, groups=groups)
        self.training_data = self.start(self.training_data_file, self.sequence_dropout, temporal=temporal, inference_type=inference_type, groups=groups)
        self.memory_data = self.start(self.memory_data_file, self.sequence_dropout, temporal=temporal, inference_type=inference_type, groups=groups)
        self.validation_data = self.start(self.validation_data_file, temporal=temporal, inference_type=inference_type, groups=groups)
        self.evaluation_data = self.start(self.evaluation_data_file, temporal=temporal, inference_type=inference_type, groups=groups)
        self.all_data = self.start(file, temporal=temporal, inference_type=inference_type, groups=groups)

    def iterate_batch(self, batch_dim, raw_batch=False):
        # Reset and shuffle batch when all items have been iterated
        if self.batch_begin > len(self.training_data) - batch_dim:
            # If sequence dropout
            if self.sequence_dropout:
                self.training_data = self.start(self.training_data_file, self.sequence_dropout)

            # Reset batch index
            self.batch_begin = 0

            # Shuffle PD data
            random.shuffle(self.training_data)

        # Index of the end boundary of this batch
        batch_end = min(self.batch_begin + batch_dim, len(self.training_data))

        # Batch
        batch = self.training_data[self.batch_begin:batch_end]

        # Update batch index
        self.batch_begin = batch_end

        # Update epoch
        self.epoch_complete = self.batch_begin > len(self.training_data) - batch_dim

        # if "time_dim" not in batch[0]:
        #     return np.stack([patient_records["inputs"] for patient_records in batch]), \
        #            np.stack([patient_records["desired_outputs"] for patient_records in batch]), \
        #            np.stack([patient_records["time_ahead"] for patient_records in batch])
        if raw_batch:
            return batch

        # Return inputs, desired outputs, and time_dims
        return np.stack([patient_records["inputs"] for patient_records in batch]), \
               np.stack([patient_records["desired_outputs"] for patient_records in batch]), \
               np.array([patient_records["time_dim"] for patient_records in batch]), \
               np.stack([patient_records["time_ahead"] for patient_records in batch])

    def start(self, file, sequence_dropout_prob=0, temporal=True, inference_type="future_score", groups=None):
        # PD data
        pd_data = []

        # List of inputs, desired outputs, and time dimensions
        for patient in file["PATNO"].unique():
            # All patient records (sorted by date-time)
            patient_records = file[file["PATNO"] == patient].sort_values(["INFODT"])

            # If sequence dropout TODO shouldn't this only be used in he batch iterator?
            if sequence_dropout_prob:
                # Patient records to use
                patient_record_indices = []

                # Sequence dropout
                for index in range(patient_records.shape[0]):
                    if random.random() >= sequence_dropout_prob or \
                            (index == patient_records.shape[0] - 2 and len(patient_record_indices) == 0) or \
                            (index == patient_records.shape[0] - 1 and len(patient_record_indices) == 1):
                        patient_record_indices.append(index)

                # # Pick a sequence length between 2 and the actual sequence length
                # sequence_length = np.random.choice(range(2, patient_records.shape[0] + 1))
                #
                # # Take a subset of the sequences of this length
                # patient_record_indices = np.random.choice(range(patient_records.shape[0]), sequence_length, False)

                # Select patient records (and explicitly sort them by their date again to be safe)
                patient_records = patient_records.iloc[patient_record_indices, :].sort_values(["INFODT"])

            # Time dimensions
            time_dim = patient_records.shape[0] - 1

            # Time ahead between inputs and desired outputs
            time_ahead = np.zeros(self.max_num_records, dtype=np.float64)
            time_ahead[:time_dim] = patient_records["AGE"][1:].values - patient_records["AGE"][:-1].values

            # Drop selected variables
            patient_records = patient_records.drop(self.to_drop, axis=1)

            # Inputs + padding
            inputs = np.zeros((self.max_num_records, self.input_dim), dtype=np.float64)
            inputs[:time_dim] = patient_records.values[:-1]

            # Desired outputs + padding (desired outputs are: next - previous)
            desired_outputs = np.zeros((self.max_num_records, self.desired_output_dim), dtype=np.float64)

            # desired_outputs[:time_dim] = patient_records[self.targets].values[1:]
            # desired_outputs[:time_dim] = patient_records[self.targets].values[1:] - patient_records[
            #                                                                             self.targets].values[:-1]

            # Desired outputs + padding (desired outputs are: next / previous)
            desired_outputs[:time_dim] = np.divide(patient_records[self.targets].values[1:],
                                                   patient_records[self.targets].values[:-1],
                                                   out=np.zeros_like(patient_records[self.targets].values[1:]),
                                                   where=patient_records[self.targets].values[:-1] != 0)
            # print(desired_outputs[:time_dim])

            # # See which changes are very big
            # for ind, i in enumerate(desired_outputs[:length]):
            #     if (np.absolute(i) > 10).any():
            #         print("\nLarge values in targets:")
            #         print("patient: {}".format(patient))
            #         print(i)
            #         print(patient_records[targets].values[1:][ind])
            #         print(patient_records[targets].values[:-1][ind])

            # Add patient records to PD data
            if temporal:
                pd_data.append({"id": patient, "inputs": inputs, "desired_outputs": desired_outputs,
                                "time_dim": time_dim, "time_ahead": time_ahead})
            else:
                if inference_type == "future_score":
                    for i, _ in enumerate(inputs):
                        # For relational reasoning
                        if groups is not None:
                            inputs = {group: patient_records[groups[group]].values[0] for group in groups}
                        else:
                            inputs = inputs[i]

                        pd_data.append({"id": patient, "inputs": inputs, "desired_outputs": desired_outputs[i],
                                        "time_ahead": time_ahead[i]})
                elif inference_type == "rop":
                    # Variables for linear regression
                    y_var = patient_records[self.targets].values
                    # x_var = np.stack([patient_records["AGE"].values] * 4, 1)
                    x_var = patient_records["AGE"].values

                    desired_outputs = np.zeros(y_var.shape[1])

                    for i in range(y_var.shape[1]):
                        desired_outputs[i], _, _, _, _ = scipy.stats.linregress(x_var, y_var[:, i])

                    # For relational reasoning
                    if groups is not None:
                        inputs = {group: patient_records[groups[group]].values[0] for group in groups}
                    else:
                        inputs = inputs[0]

                    # Linear regression
                    # desired_outputs, _, _, _, _ = scipy.stats.linregress(x_var, y_var)
                    pd_data.append({"id": patient, "inputs": inputs, "desired_outputs": desired_outputs})

        # Return pd data
        random.shuffle(pd_data)
        return pd_data

    def read(self, data, batch_size=None, time_ahead=True, time_dims_separated=False, raw_data=False):
        # Shuffle testing data TODO: why? redundant with above shuffle
        random.shuffle(data)

        # Testing batch
        data = data if batch_size is None else data[:batch_size]

        # If time dims should be separated
        if time_dims_separated:
            time_dims_separated_data = []

            # Extract individual time dims from data
            for patient_record in data:
                for i, inputs in enumerate(patient_record["inputs"]):
                    if inputs.any():
                        time_dims_separated_data.append({"inputs": [inputs],
                                                         "desired_outputs": [patient_record["desired_outputs"][i]],
                                                         "time_dim": patient_record["time_dim"],
                                                         "time_ahead": [patient_record["time_ahead"][i]]})

            # Replace data
            data = time_dims_separated_data

        if raw_data:
            return data

        # Return test data of batch size
        if time_ahead:
            return dict(inputs=np.stack([patient_records["inputs"] for patient_records in data]),
                        desired_outputs=np.stack([patient_records["desired_outputs"] for patient_records in data]),
                        time_dims=np.array([patient_records["time_dim"] for patient_records in data]),
                        time_ahead=np.stack([patient_records["time_ahead"] for patient_records in data]))
        else:
            return dict(inputs=np.stack([patient_records["inputs"] for patient_records in data]),
                        desired_outputs=np.stack([patient_records["desired_outputs"] for patient_records in data]),
                        time_dims=np.array([patient_records["time_dim"] for patient_records in data]))

    def separate_time_dims(self, data, batch_dims=None, max_time_dim=None, time_dims=None):
        # Initialize batch dims
        batch_dims = data.shape[0] if batch_dims is None else batch_dims

        # Initialize max time dim
        max_time_dim = data.shape[1] if max_time_dim is None else max_time_dim

        use_time_dims = time_dims is not None

        # Initialize time dims
        time_dims = np.full(batch_dims, max_time_dim) if time_dims is None else time_dims

        # Time dims separated
        time_dims_separated_data = []

        # For each batch
        for batch_dim in range(batch_dims):
            # For each time
            for time_dim in range(time_dims[batch_dim]):
                # If padding
                if data[batch_dim, time_dim].any() or use_time_dims:
                    # Add record
                    time_dims_separated_data.append(data[batch_dim, time_dim])
                else:
                    break

        # Return time dims separated
        return np.squeeze(np.array(time_dims_separated_data))

    def shuffle_training_memory_split(self):
        # Shuffle patients
        random.Random().shuffle(self.training_memory_data_patients)

        # Split into training and memory data
        self.training_data_patients = list(self.training_memory_data_patients[:self.train_memory_split_index])
        self.memory_data_patients = list(self.training_memory_data_patients[self.train_memory_split_index:])

        # Assigns data sets
        self.training_data_file = self.training_memory_data_file[
            self.training_memory_data_file["PATNO"].isin(self.training_data_patients)]
        self.memory_data_file = self.training_memory_data_file[
            self.training_memory_data_file["PATNO"].isin(self.memory_data_patients)]

        # Create data
        self.training_data = self.start(self.training_data_file, self.sequence_dropout)
        self.memory_data = self.start(self.memory_data_file, self.sequence_dropout)

        # Variable for iterating batches
        self.batch_begin = 0


# Count various rare events in the course of PD
def count_rare_events(data):
    print(data["NP1HALL"].describe())
    print(data["COGSTATE"].unique())

    # MDS-UPDRS item 1.2 Hallucinations and Psychosis
    visual_hallucinations = data[data["NP1HALL"] >= 1]

    # Determination of Falls item 1. Does the participant report freezing of gait occurring in the past week?
    # recurrent_falls_1 = data[data["FRZGT1W"] >= 3]
    #
    # Falls item 2. Does participant report falls occurring in the past week that were not related to freezing of gait?
    # recurrent_falls_2 = data[data["FLNFR1W"] >= 2]

    # Cognitive Categorization item 3. Which of the following categories best describes the subject’s cognitive state?
    pd_dementia = data[data["COGSTATE"] == '3S']

    #  "Patients reporting recurrent falls due to freezing of gate: {}\n"
    #  "Patients reporting recurrent falls  unrelated to freezing of gait: {}\n"
    print("Total number of patients: {}\n"
          "Patients reporting visual hallucinations: {}\n"
          "Patients experiencing PD dementia: {}".format(len(data["PATNO"].unique()),
                                                         len(visual_hallucinations["PATNO"].unique()),
                                                         # len(recurrent_falls_1["PATNO"].unique()),
                                                         # len(recurrent_falls_2["PATNO"].unique()),
                                                         len(pd_dementia["PATNO"].unique())))


# Count missing variables per variable and visit and output summary to csv
def count_missing_values(data):
    # Variables with no records per patient
    missing = pd.DataFrame(data.groupby("PATNO").apply(lambda x: x.isnull().all()).sum(axis=0),
                           columns=["% Patients With No Record For Any Visit"])

    # Percent
    missing /= float(len(data["PATNO"].unique()))

    # Sort
    missing = missing.sort_values(by="% Patients With No Record For Any Visit")

    # Output file
    missing.to_csv("Stats/all_values_missing_per_patient_variable.csv")

    print("\nMean % of patients with no record at all for a variable: {0:.0%}".format(
        missing["% Patients With No Record For Any Visit"].mean()))

    # Return missing
    return missing


# Drop
def drop(data, to_drop=None):
    # Drop patients with only one observation
    num_observations = data.groupby(["PATNO"]).size().reset_index(name='num_observations')
    patients_with_only_one_observation = num_observations.loc[num_observations["num_observations"] == 1, "PATNO"]
    data = data[~data["PATNO"].isin(patients_with_only_one_observation)]

    print("\nNumber of patients dropped (due to them only having one observation): {}".format(
        len(patients_with_only_one_observation)))

    # Count missing
    missing = count_missing_values(data)

    # Cutoff for missing values
    cutoff = 0.17

    # Drop variables with too many missing
    data = data[missing[missing["% Patients With No Record For Any Visit"] < cutoff].index.values]

    print("\nNumber of variables dropped due to too many patients without any record of them: {} ".format(
        len(missing[missing["% Patients With No Record For Any Visit"] >= cutoff])) + "(cutoff={0:.0%})".format(cutoff))

    # Drop any other specific variables
    if to_drop is not None:
        data = data.drop([x for x in to_drop if x in data.keys()], axis=1)

    print("\nNumber of variables dropped by manual selection (due to, say, duplicates, lack of statistical meaning, or "
          "just being unwanted): {}".format(len(to_drop)))

    print("\nAll dropped variables:")
    print(list(missing[missing["% Patients With No Record For Any Visit"] >= 0.6].index.values) + to_drop)

    # Return data
    return data


# Impute missing values by interpolating by patient
def impute(data):
    # Manually encode meaningful strings
    data.loc[data["tTau"] == "<80", "tTau"] = 50
    data.loc[data["pTau"] == "<8", "pTau"] = 5

    print("\nManually imputed tTau '<80' token with 50 and pTau '<8' with 5")

    # for column in data.columns.values:
    #     if data[column].isnull().any():
    #         if column in ["Serum Glucose", "ALT (SGPT)", "Serum Bicarbonate", "Albumin-QT", "Total Bilirubin",
    #                       "AST (SGOT)", "tTau", "pTau", "LAMB2(rep2)", "LAMB2(rep1)", "PSMC4(rep1)", "SKP1(rep1)",
    #                       "GAPDH(rep1)", "HSPA8(rep1)", "ALDH1A1(rep1)"]:
    #             print("\n{}:".format(column))
    #             print(data[column].value_counts())
    #
    # {"PSMC4": "Undetermined", "SKP1": "Undetermined", "LAMB2(rep1)": "Undetermined", "LAMB2(rep2)": "Undetermined",
    #  "tTao": "<80", "pTao": "<8", }

    # Variables that are numeric in nature but mixed in with strings
    coerce_to_numeric = ["Serum Glucose", "ALT (SGPT)", "Serum Bicarbonate", "Albumin-QT", "Total Bilirubin",
                         "AST (SGOT)", "tTau", "pTau", "LAMB2(rep2)", "LAMB2(rep1)", "PSMC4(rep1)", "SKP1(rep1)",
                         "GAPDH(rep1)", "HSPA8(rep1)", "ALDH1A1(rep1)"]

    print("\nManually replaced 'undetermined' token with nan")

    # Coerce to numeric those numerics mixed with strings
    for column in coerce_to_numeric:
        data[column] = pd.to_numeric(data[column], "coerce")

    # Date-time
    data["INFODT"] = pd.to_datetime(data["INFODT"])

    # Interpolation by previous or next per patient
    interpolated = data.groupby('PATNO').apply(lambda group: group.fillna(method="ffill").fillna(method="bfill"))

    # Output to file
    interpolated.to_csv("Processed/interpolated.csv")

    # Global median if still missing
    imputed = interpolated.fillna(data.median())

    # Most common occurrence for missing strings
    imputed = imputed.apply(lambda column: column.fillna(column.value_counts().index[0]))

    # Output to file
    imputed.to_csv("Processed/imputed.csv", index=False)

    # Return imputed
    return imputed


# Encode categorical values
def encode(data):
    # Ensure numbers are seen as numeric
    for column in data.columns.values:
        data[column] = pd.to_numeric(data[column], "ignore")

    # List of non-numeric variables
    variables_to_encode = [item for item in data.columns.values if item != "INFODT" and item not in list(
        data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.values)]

    print("\nVariables converted to one-hot binary dummies:")
    print(variables_to_encode)

    # Encode strings to numeric (can also use labelbinarizer for one-hot instead!) TODO
    data[variables_to_encode] = data[variables_to_encode].apply(preprocessing.LabelEncoder().fit_transform)

    # Output to file
    data.to_csv("Processed/encoded.csv", index=False)

    # Return encoded data
    return data


# Process data
def process(data, to_drop):
    # Drop
    dropped = drop(data, to_drop)

    # Impute
    imputed = impute(dropped)

    # Encode
    data = encode(imputed)

    print("\nMean sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].mean()))
    print("\nMedian sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].median()))
    print("\nMin sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].min()))
    print("\nMax sequence length: {}".format(data.groupby(["PATNO"]).count()["NP3RIGLL"].max()))
    print("\nNumber of patients: {}".format(len(data["PATNO"].unique())))
    print("\nNumber of variables: {}".format(len(data.columns.values)))


# Main method
if __name__ == "__main__":
    # Preprocessed data  TODO: on & off dose
    preprocessed = pd.read_csv("Preprocessed/preprocessed_data_treated_and_untreated_off_PD.csv")

    print("Treated and untreated off dose measurements, PD cohort")

    # Variables to drop
    variables_to_drop = ["EVENT_ID", "GENDER", "GENDER.y", "SXDT", "PDDXDT", "SXDT_x",
                         "PDDXDT_x", "BIRTHDT.x", "INFODT_2", "ENROLL_DATE",
                         "INITMDDT", "INITMDVS", "ANNUAL_TIME_BTW_DOSE_NUPDRS_y"]

    # Data processing
    # process(preprocessed, variables_to_drop)
    count_rare_events(preprocessed)
