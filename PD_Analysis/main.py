import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd
from PD_Analysis.relational_module import MHDPA
from PD_Analysis.data import read_old
import os
import argparse
import matplotlib.pyplot as plt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-saving_logging_directory', type=str, default='saving_logging')
parser.add_argument('-inference_type', type=str, default='rop')
parser.add_argument('-name', type=str, default='name')
parser.add_argument('-name_suffix', type=str, default='6')
parser.add_argument('-relational', type=str2bool, default=True)
parser.add_argument('-entity_mlp_size_1', type=int, default=32)
parser.add_argument('-entity_mlp_size_2', type=int, default=32)
parser.add_argument('-mhdpa_key_size', type=int, default=32)
parser.add_argument('-mhdpa_value_size', type=int, default=32)
parser.add_argument('-distributional', type=str2bool, default=False)
parser.add_argument('-top_k', type=int, default=3)
parser.add_argument('-sample', type=str2bool, default=True)
parser.add_argument('-uniform_sample', type=str2bool, default=False)
parser.add_argument('-aggregate_method', type=str, default="max")
parser.add_argument('-epochs', type=int, default=1000)
parser.add_argument('-episodes', type=int, default=1)
parser.add_argument('-batch_dim', type=int, default=20)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")
args.name = args.inference_type
args.saving_logging_directory = args.saving_logging_directory + "_" + args.name_suffix
# Data reader
reader = read_old.ReadPD("/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/encoded.csv",
                        targets=["UPDRS_III"],
                        train_test_split=0.7, valid_eval_split=0.33, temporal=False, inference_type=args.inference_type,
                        groups={"Demographics": ["GENDER.x", "AGE", "TIME_SINCE_DIAGNOSIS", "RAHAWOPI", "RABLACK",
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
                                                     ]})

# Inputs
groups = {}
for g in reader.groups:
    groups[g] = tf.placeholder(tf.float32, [None, len(reader.groups[g])], "Group_{}".format(g))
outcome = tf.placeholder(tf.float32, [None, len(reader.targets)], "Outcome")
# TODO: concatenate time ahead for future score inference!! DO THIS

if args.relational:
    # Represent entities
    entities_list = []
    for g in reader.groups:
        entity_mlp_size = [args.entity_mlp_size_1, args.entity_mlp_size_2]
        entity = snt.nets.mlp.MLP(output_sizes=entity_mlp_size)(groups[g])
        entities_list.append(entity)
    entities = tf.stack(entities_list, 1)
    print("entities shape", entities.shape)

    # MHDPA to get relations
    mhdpa = MHDPA()
    mhdpa(entities, key_size=args.mhdpa_key_size, value_size=args.mhdpa_value_size, num_heads=1)

    # Distributional MHDPA
    if args.distributional:
        mhdpa.keep_most_salient(top_k=args.top_k, sample=args.sample, uniform_sample=args.uniform_sample)

    # Apply MLP
    mhdpa.apply_mlp_to_relations(residual_type="add")

    # Aggregate
    relations = mhdpa.aggregate_relations(args.aggregate_method)  # Concat for distributional
else:
    relations = tf.concat([groups[g] for g in groups], axis=1)
    preds = snt.nets.mlp.MLP([128, 128, 64, 32])(relations)

# Training loss
preds = snt.nets.mlp.MLP([32, 32, outcome.shape[1]])(relations)
loss = tf.losses.mean_squared_error(outcome, preds)
# logits = snt.nets.mlp.MLP([256, 256, 256, 256, outcome.shape[1]])(relations)
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outcome, logits=logits)
# loss = tf.reduce_mean(loss)

# Accuracy
# correct = tf.equal(tf.argmax(logits, 1), outcome)
# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
accuracy = 1 / (loss + 0.001)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss,
                           global_step=global_step)  # TODO decreasing learning rate / maybe clip by a max grad norm

# TensorBoard logging
train_summary = tf.summary.merge([tf.summary.scalar("Training_Loss", loss),
                                  # tf.summary.scalar("Training_Accuracy", accuracy)])
                                  ])
valid_summary = tf.summary.merge([tf.summary.scalar("Validation_Loss", loss),
                                  ])
# tf.summary.scalar("Validation_Accuracy", accuracy)])
total_episodes = tf.get_variable("episode", [], tf.int32, tf.zeros_initializer(), trainable=False)
increment_episode = tf.assign_add(total_episodes, 1)

# Init/save/restore variables
max_accuracy = tf.get_variable("max_accuracy", [], tf.float32, tf.zeros_initializer(), trainable=False)
update_max_accuracy = tf.cond(max_accuracy < accuracy, lambda: tf.assign(max_accuracy, accuracy),
                              lambda: tf.assign(max_accuracy, max_accuracy))
max_validation_accuracy = 0
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Training
with tf.Session() as sess:
    sess.run(init)

    # TensorBoard
    path = os.getcwd()
    saving_logging_directory = path + "/" + args.saving_logging_directory + "/"
    logs = tf.summary.merge_all()
    writer = tf.summary.FileWriter(saving_logging_directory + "Logs/" + args.name + "/" + args.name_suffix + "/",
                                   sess.graph)

    # Restore any previously saved  TODO: Resume option - separate save directory for last ckpt!
    if args.saving:
        if not os.path.exists(saving_logging_directory + "Saved/" + args.name + "/"):
            os.makedirs(saving_logging_directory + "Saved/" + args.name + "/")
        if tf.train.checkpoint_exists(saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix):
            # NOTE: for some reason continuing training does not work after restoring
            if not args.slurm:
                saver.restore(sess, saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix)

    # Epochs
    epoch = 1
    episode = 1
    while epoch <= args.epochs:
        episode_loss = 0

        # Episodes
        for _ in range(args.episodes):
            # Batch
            batch = reader.iterate_batch(args.batch_dim, raw_batch=True)
            inputs = {groups[grp]: np.stack([d["inputs"][grp] for d in batch]) for grp in groups}
            inputs.update({outcome: np.stack([d["desired_outputs"] for d in batch])})

            # Train
            _, episode_loss, summary, total_episode = sess.run([train, loss, train_summary, increment_episode], inputs)

            episode += 1
            if args.logging:
                writer.add_summary(summary, total_episode)

            # Epoch complete
            if reader.epoch_complete:
                print("Epoch {} of {} complete.".format(epoch, args.epochs))
                epoch += 1
                episode = 1
                break

        # Validation accuracy
        data = reader.read(reader.validation_data, raw_data=True)
        inputs = {groups[grp]: np.stack([d["inputs"][grp] for d in data]) for grp in groups}
        inputs.update({outcome: np.stack([d["desired_outputs"] for d in data])})
        validation_loss, summary, max_acc = sess.run([loss, valid_summary, update_max_accuracy], inputs)
        if args.logging:
            writer.add_summary(summary, total_episode)

        # Save best model
        if args.saving:
            if max_acc > max_validation_accuracy:
                print("New max accuracy: {}, Validation loss: {}".format(max_acc, validation_loss))
                max_validation_accuracy = max_acc
                if max_validation_accuracy > 0:
                    saver.save(sess, saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix)

        # Print performance
        if not args.slurm:
            print('Epoch', epoch, 'of', args.epochs, 'episode', episode, 'training loss:', episode_loss,
                  'validation loss:', validation_loss)

    # TODO: decrease learning rate, max grad norm, sysargs, slurm

    # Accuracy on test tasks TODO use individual outcome losses!
    saver.restore(sess, saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix)
    accuracies = ""
    data = reader.read(reader.evaluation_data, raw_data=True)
    inputs = {groups[grp]: np.stack([d["inputs"][grp] for d in data]) for grp in groups}
    inputs.update({outcome: np.stack([d["desired_outputs"] for d in data])})
    accuracies += "{} ".format(loss.eval(inputs))
    print(accuracies)
    predicted_observed = {"Predicted": preds.eval(inputs).flatten(), "Observed": outcome.eval(inputs).flatten()}
    # print(predicted_observed["Predicted"])
    pd.DataFrame(predicted_observed).to_csv(
        "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Stats/Predicted_Observed/predicted_observed_{}_{}.csv".format(
            args.inference_type, args.name_suffix), index=False)
    plt.plot(predicted_observed["Predicted"], predicted_observed["Observed"], 'ro')
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    plt.savefig("/Users/sam/Documents/Programming/Research/PD_Analysis/data/Stats/Plots/Plot_{}_{}.png".format(
        args.inference_type, args.name_suffix))

    # Saliences
    if args.relational:
        data = reader.read(reader.all_data, raw_data=True)
        inputs = {groups[grp]: np.stack([d["inputs"][grp] for d in data]) for grp in groups}
        inputs.update({outcome: np.stack([d["desired_outputs"] for d in data])})
        saliences = mhdpa.saliences.eval(inputs)

        patient_saliences = []
        for i, d in enumerate(data):
            for j, grp in enumerate(groups):
                for k, grp2 in enumerate(groups):
                    patient_saliences.append({"PATNO": d["id"], "Group": grp, "Relation": grp2,
                                              "Salience": saliences[i][j][k]})
        pd.DataFrame(patient_saliences).to_csv(
            "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Stats/Saliences/saliences_{}_{}.csv".format(
                args.inference_type, args.name_suffix), index=False)

# TODO if slurm, option to delete saved files so as not to take up memory (nah, increase memory tho)
