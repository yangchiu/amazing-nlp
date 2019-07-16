import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

save_dir = 'cnn_mnist/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = 'MNIST_data/'


# helper functions
def init_weights(shape):
    # shape should be a 4-D tensor,
    # which represents the shape of filter
    # [filter_height, filter_width, in_channels, out_channels]

    # truncated_normal:
    # The generated values follow a normal distribution with specified mean and standard deviation,
    # except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    init_random_weights_tensor = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_weights_tensor)


def init_bias(shape):
    # shape should be a 1-D tensor,
    # which represents the # of out_channels
    init_random_bias_tensor = tf.constant(0.1, shape=shape)
    return tf.Variable(init_random_bias_tensor)


def conv2d(x, W):
    # tf.nn.conv2d(input, filter, strides=strides, padding='SAME')
    #
    # input shape = [batch_size, in_height, in_width, in_channels]
    # filter shape = [filter_height, filter_width, in_channels, out_channels]
    # strides = [batch_dimension, height_dimension, width_dimension, channel_dimension]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    # tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')
    #
    # input shape = [batch, height, width, channels]
    # ksize = window size = [batch_dimension, height_dimension, width_dimension, channel_dimension]
    # strides = [batch_dimension, height_dimension, width_dimension, channel_dimension]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(x, shape):
    # the shape should be the filter shape
    # [filter_height, filter_width, in_channels, out_channels]
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(x, W) + b)


def normal_full_layer(input_layer, size):
    # input_layer should be a flattened 2-D tensor
    # with shape [batch_size, flattened_dimension]
    #
    # size is the number of neurons in this normal_full_layer

    # the return object of .shape is type "tensorflow.python.framework.tensor_shape.Dimension"
    # should cast Dimension object to int
    input_size = int(input_layer.shape[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


if __name__ == '__main__':

    mnist = input_data.read_data_sets(os.path.join(save_dir, data_dir), one_hot=True)

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    # for dropout's keep_prob
    hold_prob = tf.placeholder(tf.float32)

    # tf.reshape() allows us to put -1 in place of i
    # and it will dynamically reshape based on the number of training samples as the training is performed.
    # [batch_size, in_height, in_width, in_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # layers

    # 6 * 6 filter, in_channels = 1, out_channels = 32
    conv_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
    conv_1_pooling = max_pool_2by2(conv_1)
    # resolution became 28 / 2 = 14 * 14

    # 6 * 6 filter, in_channels = 32, out_channels = 64
    conv_2 = convolutional_layer(conv_1_pooling, shape=[6, 6, 32, 64])
    conv_2_pooling = max_pool_2by2(conv_2)
    # resolution became 14 / 2 = 7 * 7

    # flatten the output for full layer
    conv_2_flat = tf.reshape(conv_2_pooling, [-1, 7 * 7 * 64])
    full_layer_1 = tf.nn.relu(normal_full_layer(conv_2_flat, 1024))
    full_layer_1_dropout = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_layer_1_dropout, 10)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        epochs = 5000
        batch_size = 50

        for i in range(epochs):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})

            if i % 100 == 0:

                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                acc_value = sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0})

                print(f'=> epoch {i}, Accuracy = {acc_value}')
