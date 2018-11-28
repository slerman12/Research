import tensorflow as tf
import sonnet as snt
import bAbI
from Modules import MHDPA

# Data reader
reader = bAbI.Read()

# Inputs
question = tf.placeholder(tf.int32, [None, reader.question_max_len], "Question")
supports = tf.placeholder(tf.int32, [None, reader.supports_max_num, reader.support_max_len], "Supporting")
answer = tf.placeholder(tf.int64, [None], "Answer")

question_len = tf.placeholder(tf.int32, [None], "Length_of_Question")
support_num = tf.placeholder(tf.int32, [None], "Number_of_Supporting_Sentences")
support_pos = tf.placeholder(tf.int32, [None, reader.supports_max_num], "Positions_of_Supporting_Sentences")
support_len = tf.placeholder(tf.int32, [None, reader.supports_max_num], "Lengths_of_Supporting_Sentences")

# Word embedding
word_dim = 32
words = tf.Variable(tf.random_uniform(shape=[reader.vocab_size, word_dim], minval=-1, maxval=1))

q = tf.nn.embedding_lookup(words, question)
s = tf.nn.embedding_lookup(words, supports)

# LSTM representations
with tf.variable_scope("LSTM_Question"):
    question_embed_size = 64
    lstm = tf.nn.rnn_cell.LSTMCell(question_embed_size, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
    _, state = tf.nn.dynamic_rnn(lstm, q, dtype=tf.float32, sequence_length=question_len)
    q = state.h
    q = snt.LayerNorm()(q)

with tf.variable_scope("LSTM_Supports"):
    support_embed_size = 64
    lstm = tf.nn.rnn_cell.LSTMCell(support_embed_size, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
    s = tf.reshape(s, [-1, reader.support_max_len, word_dim])
    _, state = tf.nn.dynamic_rnn(lstm, s, dtype=tf.float32, sequence_length=tf.reshape(support_len, shape=[-1]))
    s = state.h
    s = snt.LayerNorm(axis=1)(s)  # TODO check if layer norm appropriate, though I don't see why not.
    s = tf.reshape(s, [-1, reader.supports_max_num, support_embed_size])

# Concatenate question and position to each support
q = tf.tile(tf.expand_dims(q, 1), [1, reader.supports_max_num, 1])
s_pos = tf.expand_dims(tf.cast(support_pos, tf.float32), axis=2)
s = tf.concat([q, s_pos, s], axis=2)

# Mask for padded supports
supports_mask = tf.sequence_mask(support_num, reader.supports_max_num)

# MHDPA to get relations
mhdpa = MHDPA()
relations = mhdpa(s, key_size=32, value_size=32, num_heads=1, entity_mask=supports_mask)

# Aggregate relations  # TODO don't forget MLP
# mean_relation = tf.math.reduce_mean(relations, axis=1)
mean_relation = tf.math.reduce_max(relations, axis=1)

# Training loss
logits = snt.Linear(reader.vocab_size)(mean_relation)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits)
loss = tf.reduce_mean(loss)

# Accuracy
correct = tf.equal(tf.argmax(logits, 1), answer)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Training parameters
epochs = 100000
episodes = 10
batch_dim = 100

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Epochs
    epoch = 1
    while epoch < epochs:
        episode = 0
        episode_loss = 0

        # Episodes
        for _ in range(episodes):
            # Batch
            data = reader.iterate_batch(batch_dim)
            inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                      question_len: data["question_len"], support_num: data["support_num"],
                      support_pos: data["support_pos"], support_len: data["support_len"]}

            # Train
            _, episode_loss = sess.run([optimizer, loss], inputs)
            episode_loss += episode_loss
            episode += 1

            # Epoch complete
            if reader.epoch_complete:
                epoch += 1

                # Accuracy on test tasks
                data = reader.read_test(task=1)
                inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                          question_len: data["question_len"], support_num: data["support_num"],
                          support_pos: data["support_pos"], support_len: data["support_len"]}
                print('Accuracy:', accuracy.eval())

                break

        # Validation accuracy
        data = reader.read_valid(200)
        inputs = {question: data["question"], supports: data["supports"], answer: data["answer"],
                  question_len: data["question_len"], support_num: data["support_num"],
                  support_pos: data["support_pos"], support_len: data["support_len"]}
        validation_accuracy = accuracy.eval(inputs)

        # Print performance
        print('Epoch', epoch, 'of', epochs, 'episodes', episode, 'training loss:', episode_loss,
              'validation accuracy:', validation_accuracy)
