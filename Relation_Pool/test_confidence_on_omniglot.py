import tensorflow as tf
import sonnet as snt
# from Relation_Pool.relation_pool import RelationPool
from relation_pool import RelationPool
import tensorflow_datasets as tfds
import os
import argparse
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization


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
parser.add_argument('-top_k', type=int, default=3)
parser.add_argument('-sample', type=str2bool, default=True)
parser.add_argument('-uniform_sample', type=str2bool, default=False)
parser.add_argument('-aggregate_method', type=str, default="max")
parser.add_argument('-epochs', type=int, default=10000)
parser.add_argument('-episodes', type=int, default=25)
parser.add_argument('-batch_dim', type=int, default=32)
parser.add_argument('-logging', type=str2bool, default=False)
parser.add_argument('-saving', type=str2bool, default=True)
parser.add_argument('-slurm', type=str2bool, default=False)
args = parser.parse_args()
print("\n", args, "\n")

# Data reader
data, info = tfds.load("omniglot", with_info=True)
train_data, test_data = data['train'], data['test']
train_data = train_data.repeat().shuffle(10000).batch(args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
example = tf.compat.v1.data.make_one_shot_iterator(train_data).get_next()
train_images, train_labels = example["image"], example["label"]
test_data = test_data.repeat().shuffle(10000).batch(200).prefetch(tf.data.experimental.AUTOTUNE)
test = tf.compat.v1.data.make_one_shot_iterator(test_data).get_next()
test_images, test_labels = test["image"], test["label"]


def run(images, label, batch_dim):
    # Images to representations
    kernel_size = 3
    stride_size = 2
    # model = Conv2D(62, (5, 5), strides=(stride_size, stride_size),activation='relu',input_shape=(75, 75, 3), data_format='channels_last')(tf.cast(images, tf.float32))
    model = Conv2D(62, (5, 5), strides=(stride_size, stride_size),activation='relu')(tf.cast(images, tf.float32))
    # model = BatchNormalization()(model)
    model = Conv2D(62, (5, 5), strides=(stride_size, stride_size),activation='relu')(model)
    # model = BatchNormalization()(model)
    model = Conv2D(62, (kernel_size, kernel_size), strides=(stride_size, stride_size),activation='relu')(model)
    # model = BatchNormalization()(model)
    model = Conv2D(62, (3, 3), strides=(1, 1),activation='relu')(model)
    image_representations = BatchNormalization()(model)

    # Append coordinates to image representation
    N = image_representations.shape[1]
    range_ = tf.cast(tf.range(N), tf.float32)
    Y_coord = tf.reshape(tf.tile(range_[:, tf.newaxis], [1, N]), [N, N])
    X_coord = tf.tile(range_[tf.newaxis, :], [N, 1])
    X_coord = tf.tile(X_coord[tf.newaxis, :, :, tf.newaxis], [batch_dim, 1, 1, 1])
    Y_coord = tf.tile(Y_coord[tf.newaxis, :, :, tf.newaxis], [batch_dim, 1, 1, 1])
    final_concat = tf.concat([image_representations, X_coord], axis=3)
    final_concat = tf.concat([final_concat, Y_coord], axis=3)

    # Flatten image representation into entities
    entities = tf.reshape(final_concat, [-1, N*N, final_concat.shape[-1]])

    # MHDPA to get relations
    pool = RelationPool(entities=entities, k=N, initiate_pool_mode="confidence_sampling")
    relations, contexts = pool(level=N)


    def compute_error(relation_preds, desired_outputs):
        reshaped_relation_preds = tf.reshape(relation_preds, [-1, 1623])
        tiled_desired_outputs = tf.tile(desired_outputs[:, tf.newaxis], [1, relation_preds.shape[1]])
        reshaped_desired_outputs = tf.reshape(tiled_desired_outputs, [-1])
        error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshaped_desired_outputs,
                                                               logits=reshaped_relation_preds)
        return error


    # Training loss
    prediction, loss = pool._output_via_confidence_sampling(compute_error=compute_error, desired_outputs=label,
                                                            output_shape=1623)

    # Accuracy
    correct = tf.equal(tf.argmax(prediction, 1), label)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    return loss, accuracy


train_loss, train_accuracy = run(train_images, train_labels, args.batch_dim)
_, test_accuracy = run(test_images, test_labels, 200)

# Optimizer
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(train_loss)  # TODO decreasing learning rate / maybe clip by a max grad norm

# Init
init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    sess.run(init)

    # Epochs
    epoch = 1
    episode = 1
    while epoch <= args.epochs:
        episode_loss = 0

        # Episodes
        for _ in range(args.episodes):
            # Train
            _, episode_loss, train_acc = sess.run([train, train_loss, train_accuracy])

            episode_loss += episode_loss
            episode += 1

            # # Epoch complete
            # if reader.epoch_complete:
            #     print("Epoch {} of {} complete.".format(epoch, args.epochs))
            #     epoch += 1
            #     episode = 1
            #     break

        test_acc = sess.run([test_accuracy])

        # Print performance
        print('Epoch', epoch, 'of', args.epochs, 'episode', episode, 'training loss:', episode_loss,
              'training accuracy:', train_acc, 'test accuracy:', test_acc)




# TODO if slurm, option to delete saved files so as not to take up memory (nah, increase memory tho)
