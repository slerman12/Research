import tensorflow as tf
import sonnet as snt
import pandas as pd
from PD_Analysis.data import read
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
parser.add_argument('-inference_type', type=str, default='future_scores_one_to_one')
parser.add_argument('-name', type=str, default='name')
parser.add_argument('-name_suffix', type=str, default='1')
parser.add_argument('-data_file', type=str, default='data/Processed/future_scores_one_to_one_timespan_[0, 0.25]_UPDRS_III.csv')
parser.add_argument('-classification_or_regression', type=str, default='regression')
parser.add_argument('-epochs', type=int, default=5000)
parser.add_argument('-episodes', type=int, default=1)
parser.add_argument('-learning_rate', type=int, default=0.0001)
parser.add_argument('-batch_dim', type=int, default=32)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-restore', type=str2bool, default=False)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")


def mlp_regressor(inputs, outcome):
    mlp_layer_sizes = [128, 128, 64, 32, outcome.shape[1]]

    print("MLP layer sizes: ", mlp_layer_sizes)

    # initializer = snt.initializers.TruncatedNormal()

    # .1 dropout on first hidden layer
    hidden_layer_with_dropout = tf.nn.dropout(tf.nn.relu(snt.Linear(mlp_layer_sizes[0])(inputs)), keep_prob=0.9)

    preds = snt.nets.mlp.MLP(mlp_layer_sizes[1:])(hidden_layer_with_dropout)

    # Training loss
    loss = tf.losses.mean_squared_error(outcome, preds)

    accuracy = 1 / (loss + 0.001)

    return preds, loss, accuracy


def mlp_binary_classifier(inputs, outcome):
    mlp_layer_sizes = [256, 128, 64, 32, 1]

    print("MLP layer sizes: ", mlp_layer_sizes)

    # initializer = snt.initializers.TruncatedNormal()

    # .1 dropout on first hidden layer
    hidden_layer_with_dropout = tf.nn.dropout(tf.nn.relu(snt.Linear(mlp_layer_sizes[0])(inputs)), keep_prob=0.95)

    logits = snt.nets.mlp.MLP(mlp_layer_sizes[1:])(hidden_layer_with_dropout)

    # Training loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outcome, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    # Accuracy
    preds = tf.round(tf.nn.sigmoid(logits))
    correct_pred = tf.equal(preds, outcome)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return preds, loss, accuracy


def run(**params):
    # Data reader
    reader = read.ReadPD(params["data_file"],
                         targets=params["targets"], train_test_split=0.7, valid_eval_split=0.33,
                         inference_type=params["inference_type"])

    tf.reset_default_graph()

    # Inputs
    inputs = tf.placeholder(tf.float32, [None, len(reader.features)], "Inputs")
    outcome = tf.placeholder(tf.float32, [None, len(reader.targets)], "Outcome")
    # TODO: concatenate time ahead for future score inference!! DO THIS

    print("Number of features: ", len(reader.features), "\nTarget(s): ", reader.targets)

    preds, loss, accuracy = params["model"](inputs, outcome)

    # logits = snt.nets.mlp.MLP([256, 256, 256, 256, outcome.shape[1]])(relations)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outcome, logits=logits)
    # loss = tf.reduce_mean(loss)

    # Accuracy
    # correct = tf.equal(tf.argmax(logits, 1), outcome)
    # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = params["learning_rate"]
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.94, staircase=True)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # # Trainable variables
    # variables = tf.trainable_variables()
    #
    # # Get gradients of loss and clip them according to max gradient clip norm
    # gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, variables), 5)
    #
    # # Training
    # train = optimizer.apply_gradients(zip(gradients, variables))

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
        saving_logging_directory = path + "/" + args.saving_logging_directory + "_" + str(params["name_suffix"]) + "/"
        logs = tf.summary.merge_all()
        writer = tf.summary.FileWriter(saving_logging_directory + "Logs/" + params["inference_type"] + "/" + str(params["name_suffix"]) + "/",
                                       sess.graph)

        # Restore any previously saved  TODO: Resume option - separate save directory for last ckpt!
        if args.saving:
            if not os.path.exists(saving_logging_directory + "Saved/" + params["inference_type"] + "/"):
                os.makedirs(saving_logging_directory + "Saved/" + params["inference_type"] + "/")
            if tf.train.checkpoint_exists(saving_logging_directory + "Saved/" + params["inference_type"] + "/" + str(params["name_suffix"])):
                # NOTE: for some reason continuing training does not work after restoring
                if not args.slurm and args.restore:
                    saver.restore(sess, saving_logging_directory + "Saved/" + params["inference_type"] + "/" + str(params["name_suffix"]))

        # Epochs
        epoch = 1
        episode = 1
        while epoch <= params["epochs"]:
            episode_loss = 0
            training_accuracy = 0

            # Episodes
            for _ in range(args.episodes):
                # Batch
                batch = reader.iterate_batch(args.batch_dim)
                data = {inputs: batch[0], outcome: batch[1]}

                # Train
                _, episode_loss, training_accuracy, summary, total_episode = sess.run([train, loss, accuracy, train_summary, increment_episode], data)

                episode += 1
                if args.logging:
                    writer.add_summary(summary, total_episode)

                # Epoch complete
                if reader.epoch_complete:
                    print("Epoch {} of {} complete.".format(epoch, params["epochs"]))
                    epoch += 1
                    episode = 1
                    break

            # Validation accuracy
            data = {inputs: reader.validation_data[0], outcome: reader.validation_data[1]}
            validation_loss, summary, max_acc, validation_acc = sess.run([loss, valid_summary, update_max_accuracy, accuracy], data)
            if args.logging:
                writer.add_summary(summary, total_episode)

            # Save best model
            if args.saving:
                if max_acc > max_validation_accuracy:
                    print("New max validation accuracy: {}, Validation loss: {}".format(max_acc, validation_loss))
                    max_validation_accuracy = max_acc
                    if max_validation_accuracy > 0:
                        saver.save(sess, saving_logging_directory + "Saved/" + params["inference_type"] + "/" + str(params["name_suffix"]))

            # Print performance
            if not args.slurm:
                print('Epoch', epoch, 'of', params["epochs"], 'episode', episode, 'training loss:', episode_loss, "training accuracy:", training_accuracy,
                      'validation loss:', validation_loss, 'validation accuracy:', validation_acc)

        # TODO: decrease learning rate, max grad norm, sysargs, slurm


        # DELETE - just testing no validation for classification on account of class imbalances - possible solution would be to balance training and use running average validation for saving though still biasses towards long training periods that favor heavier validation class, or rather than saving based on best val, save until training acc running average gets better than validation acc running average and validation accuracy no longer improving (slope falls below 0) but this as major problems too -- no, just use a validation set consisting of balanced classes
        if params["model"] == mlp_binary_classifier:
            saver.save(sess, saving_logging_directory + "Saved/" + params["inference_type"] + "/" + str(params["name_suffix"]))

        # Accuracy on test tasks TODO use individual outcome losses!
        saver.restore(sess, saving_logging_directory + "Saved/" + params["inference_type"] + "/" + str(params["name_suffix"]))
        test_loss = ""
        data = {inputs: reader.evaluation_data[0], outcome: reader.evaluation_data[1]}
        test_loss += "{} ".format(loss.eval(data))
        test_accuracy = accuracy.eval(data)
        print("Test Loss: ", test_loss, " Test accuracy: ", test_accuracy, " Max Validation Accuracy: ", max_validation_accuracy)
        predicted_observed = {"Predicted": preds.eval(data).flatten(), "Observed": outcome.eval(data).flatten()}
        # print(predicted_observed["Predicted"])
        pd.DataFrame(predicted_observed).to_csv(
            "data/Stats/Predicted_Observed/predicted_observed_{}_{}.csv".format(
                params["inference_type"], str(params["name_suffix"])), index=False)
        plt.plot(predicted_observed["Predicted"], predicted_observed["Observed"], 'ro')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig("data/Stats/Plots/Plot_{}_{}.png".format(
            params["inference_type"], str(params["name_suffix"])))
        
        return test_loss, test_accuracy

# TODO if slurm, option to delete saved files so as not to take up memory (nah, increase memory tho)


# MSE1 = run(inference_type=args.inference_type, epochs=2, learning_rate=0.01, name_suffix=args.name_suffix, targets=["SCORE_FUTURE"],
#     data_file="/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/future_scores_one_to_one.csv")
#
# MSE2 = run(inference_type=args.inference_type, epochs=2, learning_rate=0.01, name_suffix=2, targets=["SCORE_FUTURE"],
#            data_file="/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/future_scores_bl_to_one.csv")
#
# print(MSE1, MSE2)

grid_search = [["rates_one_to_one", 5000, .00001, 0, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_classification_UPDRS_III.csv", mlp_binary_classifier],
               ["rates_one_to_one", 5000, .00001, 1, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv", mlp_regressor],
               ["rates_one_to_one", 5000, .00001, 2, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv"],
               ["rates_one_to_one", 5000, .00001, 3, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv"],
               ["rates_one_to_one", 5000, .00001, 4, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .00001, 5, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .00001, 6, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .01, 7, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv"],
               ["rates_one_to_one", 5000, .01, 8, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv"],
               ["rates_one_to_one", 5000, .01, 9, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_UPDRS_III.csv"],
               ["rates_one_to_one", 5000, .01, 10, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .01, 11, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .01, 12, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one.csv"],
               ["rates_one_to_one", 5000, .00001, 13, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .00001, 14, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .00001, 15, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .00001, 16, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .00001, 17, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .00001, 18, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 19, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 20, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 21, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_bl_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 22, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 23, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"],
               ["rates_one_to_one", 5000, .01, 24, ["RATE"], "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/rates_one_to_one_MSEADLG.csv"]]


def future_scores(bl_or_one="bl", timespan_t_="", UPDRS_III_or_MSEADLG="UPDRS_III"):
    return "future_scores_one_to_one", "SCORE_FUTURE",  "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/future_scores_{}_to_one_{}{}.csv".format(bl_or_one, timespan_t_, UPDRS_III_or_MSEADLG), "regression"


def rates(bl_or_one="bl", classification_or_regression="regression", UPDRS_III_or_MSEADLG="UPDRS_III"):
    return "rates_one_to_one", "RATE",  "/Users/sam/Documents/Programming/Research/PD_Analysis/data/Processed/future_scores_{}_to_one_{}_{}.csv".format(bl_or_one, classification_or_regression, UPDRS_III_or_MSEADLG), classification_or_regression

# results = []
# for pset in grid_search:
#     results.append(run(inference_type=pset[0], epochs=pset[1], learning_rate=pset[2], name_suffix=pset[3],
#                        targets=pset[4], data_file=pset[5], model=pset[6])[0])


results = run(inference_type=args.inference_type, epochs=args.epochs, learning_rate=args.learning_rate, name_suffix=args.name_suffix,
              targets=["SCORE_FUTURE"] if args.inference_type == "future_scores_one_to_one" else ["RATE"],
              data_file=args.data_file, model=mlp_regressor if args.classification_or_regression == "regression" else mlp_binary_classifier)

print("\n", args, "\n")
print(results)
