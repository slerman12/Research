import itertools
import json
from itertools import combinations
import ast
import pandas as pd
import numpy as np
import tensorflow as tf
from PD_Analysis.runner import mlp_regressor
import plotly.offline as plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def partial_dependency_inputs(data, features, unique_value_sample_size=1000):
    data_num_rows = len(data.index)

    unique_feature_values = data.groupby(list(set(flatten(features)))).size().reset_index()
    unique_feature_values_length = len(unique_feature_values.index)
    if unique_feature_values_length > unique_value_sample_size:
        unique_feature_values = unique_feature_values.sample(unique_value_sample_size)
        unique_feature_values_length = unique_value_sample_size
    unique_feature_values_repeated = [unique_feature_values[feature].repeat(data_num_rows) for feature in
                                      list(set(flatten(features)))]

    data = data.drop(list(set(flatten(features))), axis=1)
    data = data.iloc[np.tile(np.arange(data_num_rows), unique_feature_values_length)]

    for i, feature in enumerate(list(set(flatten(features)))):
        data[feature] = unique_feature_values_repeated[i].values

    return data


def partial_dependence(run_model, input_data, features):
    # Run model on all inputs, average outputs grouped by features
    outputs = run_model(input_data)
    partial_dependence_data = input_data[list(set(flatten(features)))].copy()
    partial_dependence_data[str(features)] = outputs
    partial_dependence_data = partial_dependence_data.groupby(list(set(flatten(features)))).mean().reset_index()
    # print(partial_dependence_data.describe())
    return partial_dependence_data


def h_statistic(partial_dependences, features):
    # Assumes all partial dependencies are in a dict with their respective feature(s) as key
    # Merge, zero center, compute h statistic
    data = partial_dependences[str(features)].copy()
    for pd in [key for key in partial_dependences if key != str(features)]:
        data = data.merge(partial_dependences[pd].copy(), how='left',
                          on=[key for key in partial_dependences[pd].columns.values if '[' not in key])
        # TODO how can it possibly merge two sets that were both sampled? Key values would not align
    # print("merged")
    # print(data.describe())

    # Zero center
    for o in list(data.columns.values):
        if o not in list(set(flatten(features))):
            data[o] = data[o] - data[o].mean()
    # print("zero centered")
    # print(data.describe())

    # Friedman H statistic
    h_stat = data[str(features)].copy()
    for pd in [key for key in partial_dependences if key != str(features)]:
        h_stat -= data[pd]
    # print("numerator subtraction")
    # print(data[str(features)].describe())
    h_stat = h_stat.pow(2)
    # print("numerator squared")
    # print(h_stat.describe())
    h_stat = h_stat.sum() / data[str(features)].pow(2).sum()
    # print("interaction effect")
    # print(h_stat)
    return h_stat


# def partial_dependency_inputs(data, features):
#     data_num_rows = len(data.index)
#
#     unique_feature_values = [data[feature].drop_duplicates() for feature in features]
#     unique_feature_values_length = [len(i) for i in unique_feature_values]
#     for i, _ in enumerate(unique_feature_values):
#         if len(unique_feature_values[i]) > 100:
#             unique_feature_values[i] = unique_feature_values[i].sample(100)
#             unique_feature_values_length[i] = 100
#     unique_feature_values_repeated = [values.repeat(data_num_rows * np.prod([length for j, length in enumerate(unique_feature_values_length) if i != j])) for i, values in enumerate(unique_feature_values)]
#
#     medians = data.median(axis=0)
#     medians = pd.DataFrame([medians.values], columns=medians.index)
#     medians_replicated = medians.iloc[[0] * data_num_rows * np.prod([length for length in unique_feature_values_length])]
#     data = medians_replicated.drop(features, axis=1)
#
#     for i, feature in enumerate(features):
#         data[feature] = unique_feature_values_repeated[i].values
#
#     return data


# center outputs at 0 (i think), average across patients (one value per feature/interaction value(s));
# perhaps these are called f values, compute h statistic, voila -- get rid of medians, only use real data interaction
# values

# def partial_dependency_data(model_func, data, interaction_size=1):
# PD_Analysis/data/Processed/future_scores_one_to_one_timespan_0_0.75_MCATOT.csv
def sublists(l):
    x = []
    for i in range(1, len(l) + 1):
        x.extend([list(y) for y in combinations(l, i)])
    return x


flatten = lambda l: [item for sublist in l for item in sublist] if isinstance(l[0], list) else l

d = pd.read_csv("data/Processed/future_scores_one_to_one_timespan_0_2_MCATOT.csv").drop(
    ["SCORE_FUTURE", "PATNO", "INFODT"], axis=1)
inputs = tf.placeholder(tf.float32, [None, len(d.columns.values)], "Inputs")
outcome = tf.placeholder(tf.float32, [None, 1], "Outcome")

preds, _, _ = mlp_regressor(inputs, outcome)

saver = tf.train.Saver()

# Training
with tf.Session() as sess:
    saver.restore(sess, 'saving_logging_name_h_stat_moca_2_yrs/' + "Saved/" + "name" + "_" + "h_stat_moca_2_yrs")


    def model(input_data):
        data = {inputs: input_data}
        return preds.eval(data).flatten()


    # all_f = []
    # f = []
    #
    # for i in range(1, 3):
    #     f.append([list(x) for x in combinations(l, i)])
    #     print(pd.read_csv("data/Processed/encoded.csv")["MCATOT"].describe())
    #     bla = partial_dependency_inputs(d, f)
    #     bla2 = {str(sub): partial_dependence(model, bla, sub) for sub in sublists(f)}
    #     bla3 = h_statistic(bla2, f)
    #     print(bla3)

    groups = {"Demographics": ["GENDER.x", "AGE", "TIME_SINCE_DIAGNOSIS", "RAHAWOPI", "RABLACK",
                               "RAASIAN", "RAINDALS", "RAWHITE", "HISPLAT", "EDUCYRS"],
              "General_Physical_Exam_And_Other_Exams": ["Skin", "Head/Neck/Lymphatic", "Eyes",
                                                        "Ears/Nose/Throat", "Lungs",
                                                        "Cardiovascular (including peripheral vascular)",
                                                        "Abdomen", "Musculoskeletal", "Neurological",
                                                        "Psychiatric", "CN346RSP", "CN5RSP", "CN2RSP",
                                                        "CN12RSP", "CN11RSP", "CN910RSP", "CN8RSP",
                                                        "CN7RSP", "MSLLRSP", "MSLARSP", "MSRLRSP",
                                                        "MSRARSP", "COHSLRSP", "COHSRRSP", "COFNRRSP",
                                                        "COFNLRSP", "SENRLRSP", "SENLARSP", "SENRARSP",
                                                        "SENLLRSP", "RFLLLRSP", "RFLRLRSP", "RFLLARSP",
                                                        "RFLRARSP", "PLRRRSP", "PLRLRSP"],
              "PD_Profile_History": ["TIME_SINCE_DIAGNOSIS", "TIME_SINCE_FIRST_SYMPTOM", "HAFSIBPD",
                                     "FULSIBPD", "BIODADPD", "BIOMOMPD", "PAGPARPD", "MATAUPD",
                                     "MAGPARPD", "PATAUPD",
                                     # "UPSITBK4", "UPSITBK3", "UPSITBK2", "UPSITBK1",  # These need to be
                                     # added together
                                     # "CNSOTH", "BRNINFM", "EPILEPSY", "DEPRS", "NARCLPSY", "RLS",
                                     # "HETRA", "STROKE", "SLPDSTRB", "DRMREMEM", "MVAWAKEN", "DRMOBJFL",
                                     # "DRMUMV", "DRMFIGHT", "PARKISM", "DRMVERBL", "SLPINJUR", "SLPLMBMV",
                                     # "DRMNOCTB", "DRMAGRAC", "DRMVIVID",
                                     # Total score generated feature
                                     # Categorical generated feature
                                     # "SCORE",
                                     "PDSURG", "DFRSKFCT", "DFPRESNT", "DFRPROG", "DFSTATIC",
                                     "DFMYOCLO", "DFBRADYA", "DFAKINES", "DFBRPLUS", "DFHEMPRK",
                                     "DFDYSTON", "DFCOGNIT", "DFPSYCH", "DFOTHPG", "DFFALLS", "DFFREEZ",
                                     "DFGAIT", "DFBRADYP", "DFAGESX", "DFURDYS", "DFRTREMP", "DFTONE",
                                     "DFUNIRIG", "DFAXRIG", "DFRIGIDA", "DFRIGIDP", "DFOTHTRM",
                                     "DFPATREM", "DFRTREMA", "DFBULBAR", "DFOTHCRS", "DFRAPSPE",
                                     "DFEYELID", "DFOCULO", "DFBWLDYS", "DFPGDIST", "DFPSHYPO",
                                     "DFHEMTRO", "DFOTHHYP", "DFOTHRIG", "DFCHOREA", "DFNEURAB",
                                     "DFOTHABR", "DFSTROKE", "DFSEXDYS", "DXTREMOR", "DOMSIDE",
                                     "DXBRADY", "DXOTHSX", "DXRIGID", "DXPOSINS"
                                     ],
              "Vital_Signs": ["SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND", "HRSTND", "WGTKG",
                              # Presence of Orthostatic Blood Pressure generated feature
                              ],
              "CSF_Biospecimens": ["CSFAlphasynuclein", "ABeta142", "Abeta42", "pTau", "pTau181P", "tTau",
                                   "Totaltau",
                                   # Ratios  "tTau / ABeta142", "tTau / CSFAlphasynuclein",
                                   # "ABeta142 / CSFAlphasynuclein", "pTau / ABeta142",
                                   # "pTau / CSFAlphasynuclein", "pTau / tTau"
                                   ],
              "Blood_Chemistry_And_Hematology_Labs": ["Monocytes", "Monocytes (%)", "Eosinophils",
                                                      "Eosinophils (%)", "Basophils", "Platelets",
                                                      "Neutrophils (%)", "Neutrophils",
                                                      "Lymphocytes (%)", "Lymphocytes", "Basophils (%)",
                                                      "Hematocrit", "RBC Morphology", "RBC",
                                                      "Hemoglobin", "WBC", "Total Bilirubin",
                                                      "Serum Glucose", "Total Protein", "Albumin-QT",
                                                      "Alkaline Phosphatase-QT", "Serum Sodium",
                                                      "Serum Potassium", "Serum Bicarbonate",
                                                      "Serum Chloride", "Calcium (EDTA)",
                                                      "Creatinine (Rate Blanked)", "ALT (SGPT)",
                                                      "AST (SGOT)", "Urea Nitrogen", "Serum Uric Acid",
                                                      "SerumIGF1"],
              "Imaging": ["CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"
                          # Right striatum and left striatum generated features
                          ],
              "Mood": ["GDSALIVE", "GDSWRTLS", "GDSENRGY", "GDSHOPLS", "GDSBETER", "GDSMEMRY", "GDSHOME",
                       "GDSHAPPY", "GDSHLPLS", "GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED",
                       "GDSGSPIR", "GDSAFRAD",
                       # Total score and depression generated features
                       "STAIAD1", "STAIAD2", "STAIAD3", "STAIAD4", "STAIAD5",
                       "STAIAD6", "STAIAD7", "STAIAD8", "STAIAD9", "STAIAD10", "STAIAD11", "STAIAD12",
                       "STAIAD13", "STAIAD14", "STAIAD15", "STAIAD16", "STAIAD17", "STAIAD18",
                       "STAIAD19", "STAIAD20", "STAIAD21", "STAIAD22", "STAIAD23", "STAIAD24",
                       "STAIAD25", "STAIAD26", "STAIAD27", "STAIAD28", "STAIAD29", "STAIAD30",
                       "STAIAD31", "STAIAD32", "STAIAD33", "STAIAD34", "STAIAD35", "STAIAD36",
                       "STAIAD37", "STAIAD38", "STAIAD39", "STAIAD40"
                       # Total score generated feature
                       ],
              "Cognition": ["MCACLCKN", "MCALION", "MCACLCKH", "MCAALTTM", "MCACUBE", "MCACLCKC",
                            "MCAVIGIL", "MCARHINO", "MCAREC4", "MCAREC5", "MCADATE", "MCAMONTH", "MCAYR",
                            "MCADAY", "MCAPLACE", "MCACITY", "MCACAMEL", "MCAREC3", "MCABDS", "MCAFDS",
                            "MCAREC2", "MCAREC1", "MCAABSTR", "MCAVF", "MCAVFNUM", "MCASNTNC", "MCASER7",
                            "MCATOT", "VLTANIM", "VLTVEG", "VLTFRUIT",
                            # Total generated feature
                            "HVLTRT1", "HVLTRT2", "HVLTRT3", "HVLTRDLY", "HVLTREC", "HVLTFPRL",
                            "HVLTFPUN", "DVT_TOTAL_RECALL", "DVT_DELAYED_RECALL", "DVT_RETENTION",
                            "DVT_RECOG_DISC_INDEX",
                            "LNS1A", "LNS1B", "LNS1C", "LNS2A", "LNS2B", "LNS2C", "LNS3A", "LNS3B",
                            "LNS3C", "LNS4A", "LNS4B", "LNS4C", "LNS5A", "LNS5B", "LNS5C", "LNS_TOTRAW",
                            "JLO_TOTRAW", "COGSTATE", "COGDECLN"
                            ],
              "Function": ["UPDRS_I", "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS",
                           "NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG",
                           "UPDRS_II", "NP2HYGN", "NP2WALK", "NP2TURN", "NP2HOBB", "NP2HWRT", "NP2DRES",
                           "NP2EAT", "NP2SWAL", "NP2SALV", "NP2TRMR", "NP2SPCH", "NP2RISE", "NP2FREZ",
                           "MSEADLG"],
              "Motor": ["UPDRS_III", "NP3RTARU", "NP3KTRML", "NP3KTRMR", "NP3PTRML", "NP3PTRMR",
                        "NP3BRADY", "NP3POSTR", "NP3PSTBL", "NP3FRZGT", "NP3GAIT", "NP3RISNG",
                        "NP3LGAGL", "NP3LGAGR", "NP3TTAPL", "NP3TTAPR", "NP3PRSPL", "NP3PRSPR",
                        "NP3HMOVL", "NP3HMOVR", "NP3RTALU", "NP3RTARL", "NP3RTALJ", "NP3RTCON",
                        "NP3RTALL", "NP3FTAPL", "NP3FTAPR",
                        # "NP3RIGRL",
                        "NP3RIGLL", "NP3RIGLU",
                        "NP3RIGRU", "NP3RIGN", "NP3FACXP", "NP3SPCH",
                        # Total sums generated features
                        "NHY"
                        ],
              "Other": ["ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8",
                        "SCAU1", "SCAU2", "SCAU3", "SCAU4", "SCAU5", "SCAU6", "SCAU7", "SCAU8", "SCAU9",
                        "SCAU10", "SCAU11", "SCAU12", "SCAU13", "SCAU14", "SCAU15", "SCAU16", "SCAU17",
                        "SCAU18", "SCAU19", "SCAU20", "SCAU21", "SCAU26B", "SCAU26C", "SCAU26D",
                        "CNTRLDSM", "TMDISMED", "TMTRWD", "TMTMTACT", "TMTORACT", "CNTRLEAT", "TMEAT",
                        "CNTRLBUY", "TMBUY", "CNTRLSEX", "TMSEX", "CNTRLGMB", "TMGAMBLE"],
              "PD_Risk_Identifiers": ["UPSITBK4", "UPSITBK3", "UPSITBK2", "UPSITBK1",
                                      # SUM not individual
                                      "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR",
                                      "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN",
                                      "DRMREMEM", "SLPDSTRB", "STROKE", "HETRA", "PARKISM", "RLS",
                                      "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH",
                                      # SUM not individual
                                      "SCORE"
                                      ]}

    cache = {"inputs": {}, "pd": {}}


    def add_to_cache(item, name, pd_or_inputs="pd"):
        cache[pd_or_inputs][name] = item
        return item


    def check_cache(runny, item, name):
        if str(name) in cache:
            return cache[str(name)]
        else:
            cache[str(name)] = partial_dependence(runny, item, name)
            return cache[str(name)]

    interaction_size = 2  # TODO 3 way interactions prob should only have two way PDs subtracted
    max_unique_value_size = 80

    write_output = "{"
    max_group_interaction = 0


    # f = list(set(flatten([groups[key] for key in groups])))
    # for interaction in combinations(f, interaction_size):
    #     interaction = list(interaction)
    #     # bla = {str(sub): add_to_cache(partial_dependency_inputs(d, sub, max_unique_value_size), str(sub), "inputs") if str(sub) not in cache["inputs"] else cache["inputs"][str(sub)] for sub in sublists(interaction)}
    #     bla = {str(sub): partial_dependency_inputs(d, sub, max_unique_value_size) for sub in sublists(interaction)}
    #     # bla2 = {str(sub): add_to_cache(partial_dependence(model, bla[str(sub)], sub), str(sub), "pd") if str(sub) not in cache["pd"] else cache["pd"][str(sub)] for sub in sublists(interaction)}
    #     bla2 = {str(sub_f): partial_dependence(model, bla[str(sub_f)], sub_f) for sub_f in sublists(interaction)}
    #     bla3 = h_statistic(bla2, interaction)
    #     print('"{}": {},'.format(str(interaction), bla3))

    # for names in combinations(groups, interaction_size):
    #     interaction = [groups[name] for name in names]
    #     # bla = {str(sub): add_to_cache(partial_dependency_inputs(d, sub, max_unique_value_size), str(sub), "inputs") if str(sub) not in cache["inputs"] else cache["inputs"][str(sub)] for sub in sublists(interaction)}
    #     bla = {str(sub): partial_dependency_inputs(d, sub, max_unique_value_size) for sub in sublists(interaction)}
    #     # bla2 = {str(sub): add_to_cache(partial_dependence(model, bla[str(sub)], sub), str(sub), "pd") if str(sub) not in cache["pd"] else cache["pd"][str(sub)] for sub in sublists(interaction)}
    #     bla2 = {str(sub): partial_dependence(model, bla[str(sub)], sub) for sub in sublists(interaction)}
    #     bla3 = h_statistic(bla2, interaction)
    #     write_output += '"{}": {},\n'.format(str(names), bla3)
    #     print('"{}": {},'.format(str(names), bla3))
    #     if bla3 > max_group_interaction:
    #         max_group_interaction = bla3
    #         group_interaction = [name for name in names]
    #
    # write_output += '}'
    ## write_output = write_output[:-1] + '}'
    # print('}')
    #
    # with open("subgroup_interactions", "w") as file:
    #     file.write(write_output)

    with open("subgroup_interactions") as f:
        data = json.load(f)
        for pair in data:
            if data[pair] > max_group_interaction:
                max_group_interaction = data[pair]
                group_interaction = ast.literal_eval(pair)

    print(group_interaction)

    cache = {"inputs": {}, "pd": {}}
    write_output = "{"

    # group_interaction = ['Blood_Chemistry_And_Hematology_Labs', 'Function']
    # interactions = list(itertools.product(groups[group_interaction[0]], groups[group_interaction[1]]))
    # for interaction in interactions:
    #     if interaction[0] != interaction[1]:
    #         interaction = list(interaction)
    #         # bla = {str(sub): add_to_cache(partial_dependency_inputs(d, sub, max_unique_value_size), str(sub), "inputs") if str(sub) not in cache["inputs"] else cache["inputs"][str(sub)] for sub in sublists(interaction)}
    #         bla = {str(sub): partial_dependency_inputs(d, sub, max_unique_value_size) for sub in sublists(interaction)}
    #         # bla2 = {str(sub): add_to_cache(partial_dependence(model, bla[str(sub)], sub), str(sub), "pd") if str(sub) not in cache["pd"] else cache["pd"][str(sub)] for sub in sublists(interaction)}
    #         bla2 = {str(sub_f): partial_dependence(model, bla[str(sub_f)], sub_f) for sub_f in sublists(interaction)}
    #         bla3 = h_statistic(bla2, interaction)
    #         print('"{}": {},'.format(str(interaction), bla3))

    max_unique_value_size = 120

    # group_interaction = ['Blood_Chemistry_And_Hematology_Labs', 'Function']
    interactions = list(itertools.product(groups[group_interaction[0]], groups[group_interaction[1]]))
    for interaction in interactions:
        if interaction[0] != interaction[1]:
            interaction = list(interaction)
            # bla = {str(sub): add_to_cache(partial_dependency_inputs(d, sub, max_unique_value_size), str(sub), "inputs") if str(sub) not in cache["inputs"] else cache["inputs"][str(sub)] for sub in sublists(interaction)}
            bla = {str(sub): partial_dependency_inputs(d, sub, max_unique_value_size) for sub in sublists(interaction)}
            # bla2 = {str(sub): add_to_cache(partial_dependence(model, bla[str(sub)], sub), str(sub), "pd") if str(sub) not in cache["pd"] else cache["pd"][str(sub)] for sub in sublists(interaction)}
            bla2 = {str(sub_f): partial_dependence(model, bla[str(sub_f)], sub_f) for sub_f in sublists(interaction)}
            bla3 = h_statistic(bla2, interaction)
            write_output += '"{}": {},\n'.format(str(interaction), bla3)
            print('"{}": {},'.format(str(interaction), bla3))

    write_output += '}'
    ## write_output = write_output[:-1] + '}'
    print('}')
    with open("feature_interactions", "w") as file:
        file.write(write_output)

    def graph(stats):
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "domain"}, {"type": "domain"}]],
                            column_widths=[5, 5], column_titles=["Top 15 Interaction Effects", "Bottom 15 Interaction Effects"])

        fig.add_trace(go.Table(header=dict(
            values=["Subgroups", "Interaction Effect"],
            align='center'),
            columnwidth=[7, 4],
            cells=dict(values=[[x.replace("_", " ").replace("(", "").replace(")", "").replace(",", ",<br>") for x in sorted([subgroup for subgroup in stats["subgroups"]], key=lambda x: stats["subgroups"][x], reverse=True)][:15],
                               sorted([stats["subgroups"][subgroup] for subgroup in stats["subgroups"]], reverse=True)[:15]],
                       height=40,
                       align=['left', 'center'],
                       format=[[None], ['.7f']])),
            row=1, col=1
        )

        fig.add_trace(go.Table(header=dict(
            values=["Subgroups", "Interaction Effect"],
            align='center'),
            columnwidth=[7, 4],
            cells=dict(values=[[x.replace("_", " ").replace("(", "").replace(")", "").replace(",", ",<br>") for x in sorted([subgroup for subgroup in stats["subgroups"]], key=lambda x: stats["subgroups"][x], reverse=True)][-15:],
                               sorted([stats["subgroups"][subgroup] for subgroup in stats["subgroups"]], reverse=True)[-15:]],
                       height=40,
                       align=['left', 'center'],
                       format=[[None], ['.7f']])),
            row=1, col=2
        )

        fig.update_layout(
            title="Interaction Effects Between Subgroups For Predicting MoCA Future Scores From Any Time Step Up To 2 Years"
        )

        plotly.plot(fig, filename='subgroup_interaction_effects.html',
                    image='png', image_filename='subgroup_interaction_effects',
                    image_height=900, image_width=1090)

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "domain"}, {"type": "domain"}]],
                            column_widths=[5, 5], column_titles=["Top 15 Interaction Effects", "Bottom 15 Interaction Effects"])

        fig.add_trace(go.Table(header=dict(
            values=["Features", "Interaction Effect"],
            align='center'),
            columnwidth=[7, 4],
            cells=dict(values=[[x.replace("_", " ").replace("[", "").replace("]", "").replace(",", ",<br>") for x in sorted([subgroup for subgroup in stats["features"]], key=lambda x: stats["features"][x], reverse=True)][:15],
                               sorted([stats["features"][subgroup] for subgroup in stats["features"]], reverse=True)[:15]],
                       height=40,
                       align=['left', 'center'],
                       format=[[None], ['.7f']])),
            row=1, col=1
        )

        fig.add_trace(go.Table(header=dict(
            values=["Features", "Interaction Effect"],
            align='center'),
            columnwidth=[7, 4],
            cells=dict(values=[[x.replace("_", " ").replace("[", "").replace("]", "").replace(",", ",<br>") for x in sorted([subgroup for subgroup in stats["features"]], key=lambda x: stats["features"][x], reverse=True)][-15:],
                               sorted([stats["features"][subgroup] for subgroup in stats["features"]], reverse=True)[-15:]],
                       height=40,
                       align=['left', 'center'],
                       format=[[None], ['.7f']])),
            row=1, col=2
        )

        fig.update_layout(
            title="Interaction Effects Between Features In “Blood Chemistry + Hematology” And “Function” Subgroups For "
                  "Predicting <br>MoCA Future Scores From Any Time Step Up To 2 Years"
        )

        plotly.plot(fig, filename='feature_interaction_effects.html',
                    image='png', image_filename='subgroup_interaction_effects',
                    image_height=900, image_width=1090)

        encoded = pd.read_csv("data/Processed/encoded.csv")
        top_ranking_feature_interactions = [x.replace("[", "").replace("]", "") for x in sorted([subgroup for subgroup in stats["features"]], key=lambda x: stats["features"][x], reverse=True)][:3]

        for inter in top_ranking_feature_interactions:
            fig = make_subplots(rows=1, cols=1,
                                # specs=[[{"type": "domain"}, {"type": "domain"}]],
                                # column_widths=[5, 5],
                                # subplot_titles=top_ranking_feature_interactions
                                )

            for row, col in itertools.product(np.arange(1, 2), np.arange(1, 2)):
                # inter = top_ranking_feature_interactions[row * col - 1]
                features = inter.replace("'", "").split(", ")

                rand_patient = np.random.choice(encoded["PATNO"].unique())

                fig.add_trace(go.Scatter(x=encoded[encoded["PATNO"] == rand_patient].groupby("TIME_SINCE_FIRST_SYMPTOM").mean().index, y=encoded.groupby("TIME_SINCE_FIRST_SYMPTOM").mean()[features[0]],
                                         # mode='lines',
                                         name=features[0]), row=int(row), col=int(col))
                fig.add_trace(go.Scatter(x=encoded[encoded["PATNO"] == rand_patient].groupby("TIME_SINCE_FIRST_SYMPTOM").mean().index, y=encoded.groupby("TIME_SINCE_FIRST_SYMPTOM").mean()[features[1]],
                                         # mode='lines+markers',
                                         name=features[1]), row=int(row), col=int(col))
                fig.add_trace(go.Scatter(x=encoded[encoded["PATNO"] == rand_patient].groupby("TIME_SINCE_FIRST_SYMPTOM").mean().index, y=encoded.groupby("TIME_SINCE_FIRST_SYMPTOM").mean()["MCATOT"],
                                         # mode='lines',
                                        name="MoCA", legendgroup="MoCA"), row=int(row), col=int(col))

            fig.update_layout(
                title="MoCA Progression W/ Top Ranking Interactions For Randomly Sampled Patient",
                xaxis_title='Time Since First Symptom (Years)')

            plotly.plot(fig, filename='MoCA_of_top_ranking_interactions_{}.html'.format(inter),
                        image='png', image_filename='subgroup_interaction_effects',
                        image_height=900, image_width=1090)


    with open("subgroup_interactions") as f:
        data1 = json.load(f)
    with open("feature_interactions") as f:
        data2 = json.load(f)
    graph({"subgroups": data1, "features": data2})
