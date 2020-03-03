import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import time


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
parser.add_argument('-name_suffix', type=str, default='h_stat_moca_2_yrs')
parser.add_argument('-data_file', type=str, default='data/Processed/future_scores_one_to_one_timespan_0_2_MCATOT.csv')
parser.add_argument('-classification_or_regression', type=str, default='regression')
parser.add_argument('-epochs', type=int, default=80)
parser.add_argument('-episodes', type=int, default=1)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-batch_dim', type=int, default=32)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-restore', type=str2bool, default=False)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")

if args.slurm:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from data import read
else:
    import tensorflow as tf
    from PD_Analysis.data import read


def mlp_regressor(inputs, outcome, dropout_proba=1):
    mlp_layer_sizes = [256, 128, 64, 32, outcome.shape[1]]

    print("MLP layer sizes: ", mlp_layer_sizes)

    # initializer = snt.initializers.TruncatedNormal()

    # .1 dropout on first hidden layer TODO NO DROPOUT AT TEST TIME
    hidden_layer_with_dropout = tf.nn.dropout(tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[0],
                                                                               kernel_initializer='truncated_normal',
                                                                               bias_initializer='zeros')(inputs)),
                                              keep_prob=dropout_proba)
    h1 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[1],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(hidden_layer_with_dropout))
    h2 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[2],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(h1))
    h3 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[3],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(h2))
    preds = tf.keras.layers.Dense(mlp_layer_sizes[4],
                                  kernel_initializer='truncated_normal',
                                  bias_initializer='zeros')(h3)

    # Training loss
    loss = tf.losses.mean_squared_error(outcome, preds)

    accuracy = 1 / (loss + 0.001)

    return preds, loss, accuracy


def mlp_classifier(inputs, outcome, num_class=3, dropout_proba=1):
    mlp_layer_sizes = [256, 128, 64, 32, num_class]

    outcome = tf.cast(tf.squeeze(outcome), tf.int64)

    print("MLP layer sizes: ", mlp_layer_sizes)

    # initializer = snt.initializers.TruncatedNormal()

    # .1 dropout on first hidden layer
    hidden_layer_with_dropout = tf.nn.dropout(tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[0],
                                                                               kernel_initializer='truncated_normal',
                                                                               bias_initializer='zeros')(inputs)),
                                              keep_prob=dropout_proba)
    h1 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[1],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(hidden_layer_with_dropout))
    h2 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[2],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(h1))
    h3 = tf.nn.relu(tf.keras.layers.Dense(mlp_layer_sizes[3],
                                          kernel_initializer='truncated_normal',
                                          bias_initializer='zeros')(h2))
    logits = tf.keras.layers.Dense(mlp_layer_sizes[4],
                                   kernel_initializer='truncated_normal',
                                   bias_initializer='zeros')(h3)

    # Training loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outcome, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    # Accuracy
    preds = tf.argmax(tf.nn.softmax(logits), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, outcome), tf.float32))

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

    dropout_prob = tf.placeholder_with_default(1.0, shape=())

    if args.classification_or_regression == "regression":
        preds, loss, accuracy = params["model"](inputs, outcome, dropout_prob)
    else:
        print(reader.data.groupby(reader.targets).count())
        num_observations_per_class = reader.training[reader.targets[0]].value_counts().values
        print(num_observations_per_class)
        reader.class_balancing()
        num_observations_per_class = reader.balanced_training[reader.targets[0]].value_counts().values
        print(num_observations_per_class)
        preds, loss, accuracy = params["model"](inputs, outcome, len(
            reader.data.groupby(reader.targets).count()), dropout_prob)  # only works for one target?
    # accuracy = 1 / (loss + 0.001)  # since validation selects best model this way

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
    max_inverse_loss = tf.get_variable("max_inverse_loss", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(),
                                       trainable=False)
    update_max_inverse_loss = tf.cond(max_inverse_loss < 1 / (loss + 0.0001), lambda: tf.assign(max_inverse_loss,
                                                                                                1 / (loss + 0.0001)),
                                      lambda: tf.assign(max_inverse_loss, max_inverse_loss))
    max_validation_inverse_loss = None
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training
    with tf.Session() as sess:
        sess.run(init)

        # TensorBoard
        path = os.getcwd()
        saving_logging_directory = path + "/" + args.saving_logging_directory + "_" + params["name"] + "_" + str(
            params["name_suffix"]) + "/"
        logs = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            saving_logging_directory + "Logs/" + params["name"] + "_" + str(params["name_suffix"]) + "/",
            sess.graph)

        # Restore any previously saved  TODO: Resume option - separate save directory for last ckpt!
        if args.saving:
            if not os.path.exists(saving_logging_directory + "Saved/" + params["inference_type"] + "/"):
                os.makedirs(saving_logging_directory + "Saved/" + params["inference_type"] + "/")
            if tf.train.checkpoint_exists(
                    saving_logging_directory + "Saved/" + params["name"] + "_" + str(params["name_suffix"])):
                # NOTE: for some reason continuing training does not work after restoring
                if not args.slurm and args.restore:
                    saver.restore(sess, saving_logging_directory + "Saved/" + params["name"] + "_" + str(
                        params["name_suffix"]))

        # Epochs
        start_time = time.time()
        epoch = 1
        episode = 1
        while epoch <= params["epochs"] and time.time() - start_time < 10 * 60:
            episode_loss = 0
            training_accuracy = 0

            # Episodes
            for _ in range(args.episodes):
                # Batch
                batch = reader.iterate_batch(args.batch_dim)
                data = {inputs: batch[0], outcome: batch[1], dropout_prob: 0.9}

                # Train
                _, episode_loss, training_accuracy, summary, total_episode = sess.run(
                    [train, loss, accuracy, train_summary, increment_episode], data)

                episode += 1
                if args.logging:
                    writer.add_summary(summary, total_episode)

                # Epoch complete
                if reader.epoch_complete:
                    if not args.slurm:
                        print("Epoch {} of {} complete.".format(epoch, params["epochs"]))
                    epoch += 1
                    episode = 1
                    break

            # Validation accuracy
            data = {inputs: reader.validation_data[0], outcome: reader.validation_data[1]}
            validation_loss, summary, max_inverse_loss_updated, validation_acc = sess.run(
                [loss, valid_summary, update_max_inverse_loss, accuracy], data)
            if args.logging:
                writer.add_summary(summary, total_episode)

            # Save best model
            if args.saving:
                if max_validation_inverse_loss is None or max_inverse_loss_updated > max_validation_inverse_loss:
                    print(
                        "New min validation loss: {}, Validation accuracy: {}".format(validation_loss, validation_acc))
                    max_validation_inverse_loss = max_inverse_loss_updated
                    # if min_validation_loss is not None:
                    saver.save(sess,
                               saving_logging_directory + "Saved/" + params["name"] + "_" + str(params["name_suffix"]))

            # Print performance
            if not args.slurm:
                print('Epoch', epoch, 'of', params["epochs"], 'episode', episode, 'training loss:', episode_loss,
                      "training accuracy:", training_accuracy,
                      'validation loss:', validation_loss, 'validation accuracy:', validation_acc)

        # TODO: decrease learning rate, max grad norm, sysargs, slurm
        # DELETE - just testing no validation for classification on account of class imbalances - possible solution would be to balance training and use running average validation for saving though still biasses towards long training periods that favor heavier validation class, or rather than saving based on best val, save until training acc running average gets better than validation acc running average and validation accuracy no longer improving (slope falls below 0) but this as major problems too -- no, just use a validation set consisting of balanced classes
        if params["model"] == mlp_classifier:
            saver.save(sess, saving_logging_directory + "Saved/" + params["name"] + "_" + str(params["name_suffix"]))

        # Accuracy on test tasks TODO use individual outcome losses!
        saver.restore(sess, saving_logging_directory + "Saved/" + params["name"] + "_" + str(params["name_suffix"]))
        data = {inputs: reader.evaluation_data[0], outcome: reader.evaluation_data[1]}
        test_loss = loss.eval(data)
        test_accuracy = accuracy.eval(data)
        print("Test Loss: ", test_loss, " Test accuracy: ", test_accuracy)
        predicted_observed = {"Predicted": preds.eval(data).flatten(), "Observed": outcome.eval(data).flatten()}
        # print(predicted_observed["Predicted"])
        pd.DataFrame(predicted_observed).to_csv(
            "data/Stats/Predicted_Observed/predicted_observed_{}_{}.csv".format(
                params["name"], str(params["name_suffix"])), index=False)
        plt.plot(predicted_observed["Predicted"], predicted_observed["Observed"], 'ro')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig("data/Stats/Plots/Plot_{}_{}.png".format(
            params["name"], str(params["name_suffix"])))

        if args.classification_or_regression != "regression":

            # Confusion matrix
            cm = confusion_matrix(predicted_observed["Observed"], predicted_observed["Predicted"])

            normalize = True

            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(args.data_file.replace("/", "_").replace("data_Processed_", "").replace(".csv", ""))
            plt.colorbar()

            # if target_names is not None:
            #     tick_marks = np.arange(len(target_names))
            #     plt.xticks(tick_marks, target_names, rotation=45)
            #     plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

            plt.savefig("data/Stats/Plots/cm/cm_{}.png".format(args.data_file.replace("/", "_").replace("data_Processed_", "").replace(".csv", "")))

        return test_loss, test_accuracy


if __name__ == '__main__':
    results = run(inference_type=args.inference_type, epochs=args.epochs, learning_rate=args.learning_rate, name=args.name,
                  name_suffix=args.name_suffix,
                  targets=["SCORE_FUTURE"] if args.inference_type == "future_scores_one_to_one" else ["RATE"],
                  data_file=args.data_file,
                  model=mlp_regressor if args.classification_or_regression == "regression" else mlp_classifier)

    print("\n", args, "\n")
    print("{} {}".format(results[0], results[1]))
