import scipy
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import scipy.special
import itertools
import numpy as np


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


# Generate future scores
def generate_future_scores_one_to_one(data, id_name, score_name, time_name, from_baseline_only=False, to_timespan=None):
    """
    Generates a dataset of future scores from each individual time step to each individual future time step
    :param data: a Pandas DataFrame of the preprocessed dataset
    :param id_name: the name of the ID feature e.g. "PATNO"
    :param score_name: the name of the outcome measure being predicted e.g. "UPDRS_III"
    :param time_name: the name of the time feature e.g. "INFODT"
    :param from_baseline_only: whether to generate from baseline only or all time points
    :param to_timespan: range of time ahead to generate to [exclusive, inclusive] e.g. [0, 0.25]
    :return: returns the new dataset
    """
    # Set features
    features = list(data.columns.values)
    new_features = ["TIME_AHEAD", "SCORE_FUTURE"]
    for feature in new_features:
        features.append(feature)

    # Remove rows without score
    data = data[data[score_name].notnull()]

    # Initialize the new dataset
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress = tqdm(desc='Generating Futures', total=len(data.index))

    # For each observation
    for index, observation in data.iterrows():
        # Get time of observation
        time_now = observation.at[time_name]

        # Get patient id
        patient_id = observation.at[id_name]

        # If not baseline, break (if from baseline only)
        if from_baseline_only and time_now != data[(data[id_name] == patient_id)][time_name].values.min():
            progress.update()
            continue

        # Get future data for specified time frame relative to this observation
        if to_timespan is None:
            futures = data[(data[id_name] == patient_id) & (data[time_name] > time_now)][[id_name, time_name, score_name]]
        else:
            futures = data[(data[id_name] == patient_id) & (data[time_name] > time_now + to_timespan[0]) &
                           (data[time_name] <= time_now + to_timespan[1])][[id_name, time_name, score_name]]
        futures.rename(columns={time_name: "TIME_FUTURE", score_name: "SCORE_FUTURE"}, inplace=True)

        # If data available
        if not futures.empty:
            # Add future information to observation
            observation_futures = futures.merge(pd.DataFrame([observation]), on=[id_name], how="left")

            # Calculate time passed
            observation_futures["TIME_AHEAD"] = observation_futures["TIME_FUTURE"] - time_now

            # Drop redundant feature
            observation_futures = observation_futures.drop(["TIME_FUTURE"], axis=1)

            # Append to the new dataset
            new_data = new_data.append(observation_futures, ignore_index=True, sort=False)

        # Update progress
        progress.update()

    # Finish progress
    progress.close()

    # Output to file
    new_data.to_csv("Processed/future_scores_{}_to_one_{}{}.csv".format("bl" if from_baseline_only else "one", "" if to_timespan is None else "timespan_" + str(to_timespan) + "_", score_name), index=False)

    # Return new dataset
    return new_data


def generate_future_scores_sequence_to_one(data, id_name, score_name, time_name):
    """
    Generates a dataset of future scores from a sequence of time steps to each individual time step.
    :param data: a Pandas DataFrame of the preprocessed dataset
    :param id_name: the name of the ID feature e.g. 'PATNO'
    :param score_name: the name of the outcome measure being predicted e.g. 'UPDRS_III'
    :param time_name: the name of the time feature e.g. 'INFODT'
    :return: returns the new dataset
    """

    def total_combinations(x):
        total = 0
        for e in x:
            for i in np.arange(1, e + 1):
                total += scipy.special.comb(e, i)
        return total

    features = list(data.columns.values)
    new_features = ['SEQUENCE', 'TIME_AHEAD', 'SCORE_FUTURE']
    for feature in new_features:
        features.append(feature)

    data = data[data[score_name].notnull()]
    new_data = pd.DataFrame(columns=features)

    total_combs = total_combinations(data.groupby(id_name).size())

    progress = tqdm(desc='Generating Futures', total=total_combs)

    sequence = 0

    for index, observation in data.iterrows():
        time_now = observation.at[time_name]
        patient_id = observation.at[id_name]

        past_obs = data[(data[id_name] == patient_id) & (data[time_name] < time_now)]
        combination_set = set()

        if not past_obs.empty:
            for num_rows in range(1, len(past_obs) + 1):
                for row_combination in itertools.combinations(past_obs[time_name], num_rows):
                    combination_set.add(row_combination)
        observation = pd.DataFrame([observation])
        observation.rename(columns={time_name: 'TIME_FUTURE', score_name: 'SCORE_FUTURE'}, inplace=True)
        observation = observation[[id_name, 'TIME_FUTURE', 'SCORE_FUTURE']]

        for combination in combination_set:
            rows = past_obs[past_obs[time_name].isin(combination)]
            complete_rows = rows.merge(observation, on=[id_name], how='left')
            complete_rows['TIME_AHEAD'] = complete_rows.TIME_FUTURE - complete_rows[time_name]
            complete_rows = complete_rows.drop(["TIME_FUTURE"], axis=1)
            complete_rows['SEQUENCE'] = sequence
            new_data = new_data.append(complete_rows, ignore_index=True, sort=False)
            sequence += 1
            progress.update()

    progress.close()

    new_data.to_csv('Processed/future_scores_sequence_to_one_{}.csv'.format(score_name), index=False)
    return new_data


# Generate rate of progressions
def generate_rates_one_to_one(data, id_name, score_name, time_name, from_baseline_only=False, classification=False):
    """
    Generates a dataset of rates of progression from each individual time step thereafter
    :param data: a Pandas DataFrame of the preprocessed dataset
    :param id_name: the name of the ID feature e.g. "PATNO"
    :param score_name: the name of the outcome measure whose rate of change is being predicted e.g. "UPDRS_III"
    :param time_name: the name of the time feature e.g. "INFODT"
    :param from_baseline_only: whether to generate from baseline only or all time points
    :param classification: whether to generate classes rather than continuous
    :return: returns the new dataset
    """
    # Set features
    features = list(data.columns.values)
    new_features = ["RATE_TIMESPAN", "RATE_COUNT", "RATE"]
    for feature in new_features:
        features.append(feature)

    # Remove rows without score
    data = data[data[score_name].notnull()]

    # Initialize the new dataset
    new_data = pd.DataFrame(columns=features)

    # Initialize progress measures
    progress = tqdm(desc='Generating Rates', total=len(data.index))

    # For each observation
    for index, observation in data.iterrows():
        # Get time of observation
        time_now = observation.at[time_name]

        # Get patient id
        patient_id = observation.at[id_name]

        # If not baseline, break (if from baseline only)
        if from_baseline_only and time_now != data[(data[id_name] == patient_id)][time_name].values.min():
            progress.update()
            continue

        # Get future data for specified time frame relative to this observation
        futures = data[(data[id_name] == patient_id) & (data[time_name] >= time_now)][[time_name, score_name]]

        # If data exceeds minimum number of observations for rate
        if len(futures) >= 3:
            # Compute the rate of progression
            y_var = futures[score_name].values
            x_var = futures[time_name].values
            rate, _, _, _, _ = scipy.stats.linregress(x_var, y_var)

            # Compute the timespan from which the rate will be computed
            timespan = max(futures[time_name].values) - time_now

            # Compute the number of observations from which the rate will be computed
            count = len(futures)

            # Add new features to observation
            observation_rate = pd.DataFrame([observation])
            observation_rate["RATE"] = rate
            observation_rate["RATE_TIMESPAN"] = timespan
            observation_rate["RATE_COUNT"] = count

            # Append to the new dataset
            new_data = new_data.append(observation_rate, ignore_index=True, sort=False)

        # Update progress
        progress.update()

    # Finish progress
    progress.close()

    # Create classes (if classification)
    if classification:
        median_rate = new_data["RATE"].median()
        print("Median rate classification threshold: ", median_rate)
        print(new_data["RATE"].describe())
        new_data["RATE_CLASS"] = -1000
        new_data.loc[new_data["RATE"] < median_rate, ["RATE_CLASS"]] = 0
        new_data.loc[new_data["RATE"] >= median_rate, ["RATE_CLASS"]] = 1
        new_data["RATE"] = new_data["RATE_CLASS"]
        new_data.drop(["RATE_CLASS"], axis=1, inplace=True)
        print(new_data["RATE"].value_counts())

        # Output to file
    new_data.to_csv("Processed/rates_{}_to_one_{}_{}.csv".format("bl" if from_baseline_only else "one", "classification" if classification else "regression", score_name), index=False)

    # Return new dataset
    return new_data


# Main method
if __name__ == "__main__":
    # Preprocessed data  TODO: on & off dose
    # preprocessed = pd.read_csv("Preprocessed/preprocessed_data_treated_and_untreated_off_PD.csv")
    #
    # print("Treated and untreated off dose measurements, PD cohort")
    #
    # # Variables to drop
    # variables_to_drop = ["EVENT_ID", "GENDER", "GENDER.y", "SXDT", "PDDXDT", "SXDT_x",
    #                      "PDDXDT_x", "BIRTHDT.x", "INFODT_2", "ENROLL_DATE",
    #                      "INITMDDT", "INITMDVS", "ANNUAL_TIME_BTW_DOSE_NUPDRS_y"]

    # Data processing
    # process(preprocessed, variables_to_drop)
    # count_rare_events(preprocessed)
    # bla = pd.DataFrame({"PATNO": [0,0,0,0,0,0,1,1,1,1,1], "FEATURES": [5,5,5,5,5,5,5,5,5,5,5],
    # "AGE": [0,1,2,3,4,5,0,1,2,3,4], "UPDRS_III": [10,11,12,13,14,15,16,17,18,19,20]})

    encoded = pd.read_csv("Processed/encoded.csv")
    # future_scores = generate_future_scores_one_to_one(encoded, "PATNO", "MSEADLG", "AGE")
    # future_scores = generate_rates_one_to_one(encoded, "PATNO", "UPDRS_III", "AGE", from_baseline_only=True, classification=True)

    c = 0

    for f_bl in [True, False]:
        for to_tspan in [None, [0, .25], [0, 0.5], [0, 0.75]]:
            for targ in ["UPDRS_III", "MSEADLG"]:
                generate_future_scores_one_to_one(encoded, "PATNO", targ, "AGE", from_baseline_only=f_bl, to_timespan=to_tspan)
                c += 1

    for f_bl in [True, False]:
        for c_or_r in [True, False]:
            for targ in ["UPDRS_III", "MSEADLG"]:
                generate_rates_one_to_one(encoded, "PATNO", targ, "AGE", from_baseline_only=f_bl, classification=c_or_r)
                c += 1

                # Generalizability categorical (heuristic)
                g = 0
                if f_bl:
                    g += 7
                else:
                    g += 10

    print("Datasets: ", c)

    # generate_rates_one_to_one(encoded, "PATNO", "MSEADLG", "AGE", from_baseline_only=True, classification=True)


