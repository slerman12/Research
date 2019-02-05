import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd
from PD_Analysis.relational_module import MHDPA
from PD_Analysis.data import PPMI
import os
import argparse


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
parser.add_argument('-name_suffix', type=str, default='name_suffix')
parser.add_argument('-relational', type=str2bool, default=True)
parser.add_argument('-entity_mlp_size_1', type=int, default=8)
parser.add_argument('-entity_mlp_size_2', type=int, default=8)
parser.add_argument('-mhdpa_key_size', type=int, default=8)
parser.add_argument('-mhdpa_value_size', type=int, default=8)
parser.add_argument('-distributional', type=str2bool, default=False)
parser.add_argument('-top_k', type=int, default=3)
parser.add_argument('-sample', type=str2bool, default=True)
parser.add_argument('-uniform_sample', type=str2bool, default=False)
parser.add_argument('-aggregate_method', type=str, default="max")
parser.add_argument('-epochs', type=int, default=10000)
parser.add_argument('-episodes', type=int, default=100)
parser.add_argument('-batch_dim', type=int, default=20)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")
args.name = args.inference_type
# Data reader
reader = PPMI.ReadPD("/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/encoded.csv",
                     targets=["UPDRS_I", "UPDRS_II", "UPDRS_III", "MSEADLG"],
                     train_test_split=0.7, valid_eval_split=0.33, temporal=False, inference_type=args.inference_type,
                     groups={"Demographics": ["GENDER.x", "AGE", "TIME_SINCE_DIAGNOSIS"],
                             "Vital_Signs": ["SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND", "HRSTND"],
                             "Mood": ["GDSALIVE", "GDSWRTLS", "GDSENRGY", "GDSHOPLS", "GDSBETER", "GDSMEMRY", "GDSHOME",
                                      "GDSHAPPY", "GDSHLPLS", "GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED",
                                      "GDSGSPIR", "GDSAFRAD", "STAIAD1", "STAIAD2", "STAIAD3", "STAIAD4", "STAIAD5",
                                      "STAIAD6", "STAIAD7", "STAIAD8", "STAIAD9", "STAIAD10", "STAIAD11", "STAIAD12",
                                      "STAIAD13", "STAIAD14", "STAIAD15", "STAIAD16", "STAIAD17", "STAIAD18",
                                      "STAIAD19", "STAIAD20", "STAIAD21", "STAIAD22", "STAIAD23", "STAIAD24",
                                      "STAIAD25", "STAIAD26", "STAIAD27", "STAIAD28", "STAIAD29", "STAIAD30",
                                      "STAIAD31", "STAIAD32", "STAIAD33", "STAIAD34", "STAIAD35", "STAIAD36",
                                      "STAIAD37", "STAIAD38", "STAIAD39", "STAIAD40"],
                             "Motor": ["UPDRS_III", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "PN3RIGRL", "NP3RIGLL",
                                       "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU",
                                       "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"],
                             "DaTScan": ["CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"],
                             "Biospecimens": ["CSFAlphasynuclein", "ABeta142"],
                             "Function": ["UPDRS_I", "MSEADLG"],
                             "Cognition": ["MCATOT", "VLTANIM", "VLTVEG", "VLTFRUIT"]})

# Inputs
groups = {}
for g in reader.groups:
    groups[g] = tf.placeholder(tf.float32, [None, len(reader.groups[g])], "Group_{}".format(g))
outcome = tf.placeholder(tf.float32, [None, len(reader.targets)], "Outcome")
# TODO: concatenate time ahead for future score inference

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
    preds = snt.nets.mlp.MLP([8, 8, 8, 8])(relations)

# Training loss
preds = snt.nets.mlp.MLP([8, 8, outcome.shape[1]])(relations)
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
train = optimizer.minimize(loss, global_step=global_step)  # TODO decreasing learning rate / maybe clip by a max grad norm

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
    writer = tf.summary.FileWriter(saving_logging_directory + "Logs/" + args.name + "/" + args.name_suffix + "/", sess.graph)

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

            episode_loss += episode_loss
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
                print("New max accuracy: {}".format(max_acc))
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
        pd.DataFrame(patient_saliences).to_csv("/Users/sam/Documents/Programming/Research/PD_Analysis/data/Stats/saliences_{}.csv".format(args.inference_type), index=False)


# TODO if slurm, option to delete saved files so as not to take up memory (nah, increase memory tho)
