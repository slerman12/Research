import tensorflow as tf
import sonnet as snt
from Relation_Pool.relation_pool_sleek import RelationPool
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bAbI_Dataset import bAbI
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
parser.add_argument('-name', type=str, default='name')
parser.add_argument('-name_suffix', type=str, default='name_suffix')
parser.add_argument('-max_supporting', type=int, default=20)
parser.add_argument('-word_dim', type=int, default=20)
parser.add_argument('-question_embed_size', type=int, default=31)
parser.add_argument('-support_embed_size', type=int, default=32)
parser.add_argument('-mhdpa_key_size', type=int, default=64)
parser.add_argument('-mhdpa_value_size', type=int, default=64)
parser.add_argument('-distributional', type=str2bool, default=True)
parser.add_argument('-top_k', type=int, default=10)
parser.add_argument('-sample', type=str2bool, default=True)
parser.add_argument('-uniform_sample', type=str2bool, default=False)
parser.add_argument('-aggregate_method', type=str, default="max")
parser.add_argument('-epochs', type=int, default=10000)
parser.add_argument('-episodes', type=int, default=10)
parser.add_argument('-batch_dim', type=int, default=32)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=False)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")

# Data reader
path = os.getcwd() + "/../bAbI_Dataset/tasks_1-20_v1-2/en-valid-10k"
reader = bAbI.Read(directory=path,
                   max_supporting=args.max_supporting)

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

mode = "confidence"
aggregate_preds = False
print(mode)

inference_module = snt.nets.mlp.MLP([64, 64, reader.vocab_size])
context_predictor = snt.nets.mlp.MLP([64, reader.vocab_size])

# expert_generator = snt.nets.mlp.MLP([64 * 64, 64 * 64])
# context_query = snt.nets.mlp.MLP([64])

expert_generator = snt.nets.mlp.MLP([64 * 64, 64 * 64])
context_query = snt.nets.mlp.MLP([64])

alt_inference_module = snt.nets.mlp.MLP([64, 64, reader.vocab_size])


# Training loss
def error(relations_or_contexts, mode="relations", confidences=None, inference_func=None):
    # Predictor
    if inference_func is not None:
        predictor = alt_inference_module
    elif mode == "relations":
        predictor = inference_module
    elif mode == "contexts":
        predictor = snt.nets.mlp.MLP([64, reader.vocab_size])
    # elif mode == "experts":
    #     def expert_func(rel):
    #         e = snt.BatchApply(expert_generator)(rel)
    #         e = tf.reshape(e, shape=[-1, e.shape[1], 64, 64])
    #         dot = tf.einsum('ijkl,il->ijk', e, context_query(global_context))
    #         return snt.BatchApply(inference_module)(dot)

    # Predictions
    # if mode != "experts":
    pred = snt.BatchApply(predictor)(relations_or_contexts)
    # else:
    #     pred = expert_func(relations_or_contexts)

    # Reshaped predictions and desired outputs for error computation
    if confidences is None:
        pred_ = tf.reshape(pred, [-1, reader.vocab_size])
        desired_outputs = tf.reshape(tf.tile(answer[:, tf.newaxis], [1, pred.shape[1]]), [-1])
    else:
        pred_ = tf.nn.softmax(pred)
        # Should then normalize and use regular cross entropy
        pred_ = tf.reduce_sum(pred_ * tf.tile(confidences[:, :, tf.newaxis], [1, 1, reader.vocab_size]), axis=1)
        desired_outputs = answer

    # Errors
    errors = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=desired_outputs, logits=pred_)

    # Return errors in original shape
    if confidences is None:
        return tf.reshape(errors, [-1, pred.shape[1]])
    else:
        return errors


# error_module = snt.Module(error)

# Initiate pool
pool = RelationPool(k=10, level=5, error_func=error, mode=mode, context_aggregation="lstm",
                    aggregate_preds=False, generate_experts=False)

# Run relation pool module
relations, weights = pool(entities)

# If aggregating predictions
if aggregate_preds:
    # Aggregate predictions
    prediction = pool.aggregate_outputs(relations, weights, inference_module)
elif pool._generate_experts:
    batch_range = tf.range(tf.shape(relations)[0])
    first_index_of_each_batch = tf.stack([batch_range, tf.zeros(tf.shape(relations)[0], tf.int32)], axis=1)
    relation = tf.gather_nd(relations, first_index_of_each_batch)


    # def expert(rel):
    #     e = tf.reshape(expert_generator(rel), shape=[-1, 64, 64])
    #     dot = tf.einsum('ijk,ik->ij', e, context_query(pool._global_context))
    #     return inference_module(dot)
    def expert(rel):
        e = tf.reshape(pool._expert_generator(rel), shape=[-1, 64, 64])
        component_knowledge = tf.einsum('ijk,ik->ij', e, pool._context_query)
        return alt_inference_module(component_knowledge)


    prediction = expert(relation)
else:
    # Prediction
    prediction = pool.first_relation_output(relations, inference_module)

# Loss
loss = pool.loss

# Accuracy
correct = tf.equal(tf.argmax(prediction, 1), answer)
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
                print("Epoch {} of {} complete.".format(epoch, args.epochs))
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
        if max_acc > max_validation_accuracy:
            print("New max accuracy: {}".format(max_acc))
            max_validation_accuracy = max_acc
            if max_validation_accuracy > 0:
                if args.saving:
                    saver.save(sess, saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix)

        # Print performance
        if not args.slurm:
            print('Epoch', epoch, 'of', args.epochs, 'episode', episode, 'training loss:', episode_loss,
                  'validation accuracy:', validation_accuracy)

    # TODO: decrease learning rate, max grad norm, sysargs, slurm

    # Accuracy on test tasks
    saver.restore(sess, saving_logging_directory + "Saved/" + args.name + "/" + args.name_suffix)
    accuracies = ""
    valid_accs = ""
    for task in range(1, 21):
        data = reader.read_test(task=task)
        inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                  question_len: data["question_len"], support_num: data["support_num"],
                  support_pos: data["support_pos"], support_len: data["support_len"]}
        accuracies += "{} ".format(accuracy.eval(inputs))

        data = reader.read_valid(task=task)
        inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                  question_len: data["question_len"], support_num: data["support_num"],
                  support_pos: data["support_pos"], support_len: data["support_len"]}
        valid_accs += "{} ".format(accuracy.eval(inputs))
    print(accuracies)
    print(valid_accs)

# TODO if slurm, option to delete saved files so as not to take up memory (nah, increase memory tho)
