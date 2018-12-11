import os
import tensorflow as tf
import sonnet as snt
import bAbI
from Modules import MHDPA
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
parser.add_argument('-max_supporting', type=int, default=20)
parser.add_argument('-word_dim', type=int, default=20)
parser.add_argument('-question_embed_size', type=int, default=64)
parser.add_argument('-support_embed_size', type=int, default=64)
parser.add_argument('-mhdpa_key_size', type=int, default=64)
parser.add_argument('-mhdpa_value_size', type=int, default=64)
parser.add_argument('-distributional', type=str2bool, default=True)
parser.add_argument('-top_k', type=int, default=3)
parser.add_argument('-sample', type=str2bool, default=True)
parser.add_argument('-uniform_sample', type=str2bool, default=False)
parser.add_argument('-aggregate_method', type=str, default="max")
parser.add_argument('-epochs', type=int, default=10000)
parser.add_argument('-episodes', type=int, default=10)
parser.add_argument('-batch_dim', type=int, default=100)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()

# Data reader
reader = bAbI.Read(max_supporting=args.max_supporting)

# Inputs
question = tf.placeholder(tf.int32, [None, reader.question_max_len], "Question")
supports = tf.placeholder(tf.int32, [None, reader.supports_max_num, reader.support_max_len], "Supporting")
answer = tf.placeholder(tf.int64, [None], "Answer")

question_len = tf.placeholder(tf.int32, [None], "Length_of_Question")
support_num = tf.placeholder(tf.int32, [None], "Number_of_Supporting_Sentences")
support_pos = tf.placeholder(tf.int32, [None, reader.supports_max_num], "Positions_of_Supporting_Sentences")
support_len = tf.placeholder(tf.int32, [None, reader.supports_max_num], "Lengths_of_Supporting_Sentences")

# Word embedding
words = tf.Variable(tf.random_uniform(shape=[reader.vocab_size, args.word_dim], minval=-1, maxval=1))

q = tf.nn.embedding_lookup(words, question)
s = tf.nn.embedding_lookup(words, supports)

# LSTM representations
with tf.variable_scope("LSTM_Question"):
    lstm = tf.nn.rnn_cell.LSTMCell(args.question_embed_size, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
    _, state = tf.nn.dynamic_rnn(lstm, q, dtype=tf.float32, sequence_length=question_len)
    q = state.c
    q = snt.LayerNorm()(q)

with tf.variable_scope("LSTM_Supports"):
    lstm = tf.nn.rnn_cell.LSTMCell(args.support_embed_size, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
    s = tf.reshape(s, [-1, reader.support_max_len, args.word_dim])
    _, state = tf.nn.dynamic_rnn(lstm, s, dtype=tf.float32, sequence_length=tf.reshape(support_len, shape=[-1]))
    s = state.c
    s = snt.LayerNorm(axis=1)(s)  # TODO check if layer norm appropriate, though I don't see why not.
    s = tf.reshape(s, [-1, reader.supports_max_num, args.support_embed_size])

# Concatenate question and position to each support
q = tf.tile(tf.expand_dims(q, 1), [1, reader.supports_max_num, 1])
s_pos = tf.expand_dims(tf.cast(support_pos, tf.float32), axis=2)
entities = tf.concat([q, s_pos, s], axis=2)

# Mask for padded supports
entity_mask = tf.sequence_mask(support_num, reader.supports_max_num)

# MHDPA to get relations
mhdpa = MHDPA()
mhdpa(entities, key_size=args.mhdpa_key_size, value_size=args.mhdpa_value_size, num_heads=1, entity_mask=entity_mask)

# Distributional MHDPA
if args.distributional:
    mhdpa.keep_most_salient(top_k=args.top_k, sample=args.sample, uniform_sample=args.uniform_sample)

# Apply MLP
mhdpa.apply_mlp_to_relations()

# Aggregate
relations = mhdpa.aggregate_relations(args.aggregate_method)  # Concat for distributional

# Training loss
logits = snt.nets.mlp.MLP([256, 256, 256, 256, reader.vocab_size])(relations)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits)
loss = tf.reduce_mean(loss)

# Accuracy
correct = tf.equal(tf.argmax(logits, 1), answer)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

# Optimizer
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)  # TODO decreasing learning rate / maybe clip by a max grad norm

# TensorBoard logging
train_summary = tf.summary.merge([tf.summary.scalar("Training_Loss", loss),
                                  tf.summary.scalar("Training_Accuracy", accuracy)])
valid_summary = tf.summary.merge([tf.summary.scalar("Validation_Loss", loss),
                                  tf.summary.scalar("Validation_Accuracy", accuracy)])
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
    directory_name = "directory"
    path = os.getcwd()
    log_directory = path + "/" + directory_name + "/"
    logs = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_directory + "Logs/", sess.graph)

    # Restore any previously saved
    if args.saving:
        if not os.path.exists(log_directory + "Saved/"):
            os.makedirs(log_directory + "Saved/")
        if tf.train.checkpoint_exists(log_directory + "Saved/model"):
            # NOTE: for some reason continuing training does not work after restoring
            saver.restore(sess, log_directory + "Saved/model")

    # Epochs
    epoch = 1
    episode = 1
    while epoch < args.epochs:
        episode_loss = 0

        # Episodes
        for _ in range(args.episodes):
            # Batch
            data = reader.iterate_batch(args.batch_dim)
            inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                      question_len: data["question_len"], support_num: data["support_num"],
                      support_pos: data["support_pos"], support_len: data["support_len"]}

            # Train
            _, episode_loss, summary, total_episode = sess.run([train, loss, train_summary, increment_episode], inputs)

            episode_loss += episode_loss
            episode += 1
            if args.logging:
                writer.add_summary(summary, total_episode)

            # Epoch complete
            if reader.epoch_complete:

                epoch += 1
                episode = 1
                break

        # Validation accuracy
        data = reader.read_valid(200)
        inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                  question_len: data["question_len"], support_num: data["support_num"],
                  support_pos: data["support_pos"], support_len: data["support_len"]}
        validation_accuracy, summary, max_acc = sess.run([accuracy, valid_summary, update_max_accuracy], inputs)
        if args.logging:
            writer.add_summary(summary, total_episode)

        # Save best model
        if args.saving:
            if max_acc > max_validation_accuracy:
                max_validation_accuracy = max_acc
                if max_validation_accuracy > 0:
                    saver.save(sess, log_directory + "Saved/model")

        # Print performance
        if not args.slurm:
            print('Epoch', epoch, 'of', args.epochs, 'episode', episode, 'training loss:', episode_loss,
                  'validation accuracy:', validation_accuracy)

    # TODO: decrease learning rate, max grad norm, sysargs, slurm

    # Accuracy on test tasks
    accuracies = ""
    for task in range(1, 21):
        data = reader.read_test(task=task)
        inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                  question_len: data["question_len"], support_num: data["support_num"],
                  support_pos: data["support_pos"], support_len: data["support_len"]}
        accuracies += accuracy.eval(inputs) + " "
    print(accuracies)

