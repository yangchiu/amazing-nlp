import pickle
import numpy as np
import tensorflow as tf

cifar_dir = 'cifar-10-batches-py'
cifar_filenames = ['batches.meta', 'data_batch_1',
                   'data_batch_2', 'data_batch_3',
                   'data_batch_4', 'data_batch_5',
                   'test_batch']

class CifarHelper():

    def __init__(self, cifar_dir, cifar_filenames):

        self.i = 0

        self.train_batch = []
        self.test_batch = []

        self.label_names = None

        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None

        self.get_data(cifar_dir, cifar_filenames)
        self.setup_images_and_labels()

    def get_data(self, cifar_dir, cifar_filenames):
        print(f'* calling {CifarHelper.get_data.__name__}')

        def unpickle(filename):
            with open(filename, 'rb') as f:
                cifar_data = pickle.load(f, encoding='bytes')
            return cifar_data

        batch_meta = unpickle(f'{cifar_dir}/{cifar_filenames[0]}')
        self.label_names = batch_meta[b'label_names']
        print(f'=> get cifar label names:')
        print(f'   {self.label_names}')

        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[1]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[2]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[3]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[4]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[5]}'))
        print(f'=> get cifar train batch:')
        print(f'   {self.train_batch[0].keys()}')

        self.test_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[6]}'))
        print(f'=> get cifar test batch:')
        print(f'   {self.test_batch[0].keys()}')

    def setup_images_and_labels(self):
        print(f'* calling {CifarHelper.setup_images_and_labels.__name__}')

        print(f'======== setting up training images and labels =========')
        # train_batch[i][b'data'] is of type "numpy.ndarray" with shape (10000, 3072)
        # train_batch[i][b'labels'] is of type "list" with length 10000

        # concat 5 batches data of shape (10000, 3072) using vstack
        # the output is of shape (50000, 3072)
        self.training_images = np.vstack([batch[b'data'] for batch in self.train_batch])
        print(f'=> shape of training_images: {self.training_images.shape}')
        self.training_images = self.training_images.reshape(
            int(self.training_images.shape[0]),
            3,
            32,
            32
        ).transpose(0, 2, 3, 1) # the order should be (batch_size, height, width, channels)
        print(f'=> reshape training_images to: {self.training_images.shape}')

        print(f'=> training image max = {self.training_images.max()}, min = {self.training_images.min()}')
        self.training_images = self.training_images / 255
        print(f'=> scaling training image to max = {self.training_images.max()}, min = {self.training_images.min()}')

        # concat 5 batches labels of length 10000 using hstack
        # the output is of shape (50000,)
        self.training_labels = np.hstack([batch[b'labels'] for batch in self.train_batch])
        print(f'=> shape of training_labels: {self.training_labels.shape}')
        self.training_labels = self.one_hot_encode(self.training_labels, 10)
        print(f'=> shape of one-hot encoded training_labels: {self.training_labels.shape}')

        print(f'======== setting up test images and labels ========')

        self.test_images = np.vstack([batch[b'data'] for batch in self.test_batch])
        print(f'=> shape of test_images: {self.test_images.shape}')
        self.test_images = self.test_images.reshape(
            int(self.test_images.shape[0]),
            3,
            32,
            32
        ).transpose(0, 2, 3, 1) / 255
        print(f'=> reshape test_images to: {self.test_images.shape}')
        self.test_labels = self.one_hot_encode(
            np.hstack([batch[b'labels'] for batch in self.test_batch]),
            10
        )
        print(f'=> shape of one-hot encoded test_labels: {self.test_labels.shape}')

    def one_hot_encode(self, vector, classes=10):
        print(f'* calling {CifarHelper.one_hot_encode.__name__}')

        length = len(vector)
        output = np.zeros((length, classes))
        output[range(length), vector] = 1
        return output

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size]
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

def init_weights(shape):
    init_random_tensor = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_tensor)


def init_bias(shape):
    init_bias_tensor = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_tensor)


def con2d(x, W):
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


def convolutional_layer(input, shape):
    # the shape should be the filter shape
    # [filter_height, filter_width, in_channels, out_channels]
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(con2d(input, W) + b)


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
    cifar = CifarHelper(cifar_dir, cifar_filenames)

    # create model
    print(f'======== create model ========')

    # placeholders
    # x shape [None, 32, 32, 3]
    # y shape [None, 10]
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    # the probability for the dropout,
    # no need for shape, it just hold a single value
    hold_prob = tf.placeholder(tf.float32)

    # create layers
    # using 4 x 4 filer, input_channels = 3, output_channels = 32
    conv_1 = convolutional_layer(x, shape=[4, 4, 3, 32])
    conv_1_pooling = max_pool_2by2(conv_1)
    # output shape = [None, 16, 16, 32]

    conv_2 = convolutional_layer(conv_1_pooling, shape=[4, 4, 32, 64])
    conv_2_pooling = max_pool_2by2(conv_2)
    # output shape = [None, 8, 8, 64]

    conv_2_flat = tf.reshape(conv_2_pooling, [-1, 8 * 8 * 64])
    full_layer_1 = tf.nn.relu(normal_full_layer(conv_2_flat, 1024))
    full_layer_1_dropout = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_layer_1_dropout, 10)

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:
        print(f'======== start session ========')
        sess.run(init)

        epochs = 5000
        batch_size = 100

        for i in range(epochs):
            batch = cifar.next_batch(batch_size)
            sess.run(train, feed_dict={
                x: batch[0],
                y_true: batch[1],
                hold_prob: 0.5
            })

            if i % 100 == 0:
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                acc_value = sess.run(acc, feed_dict={
                    x: cifar.test_images,
                    y_true: cifar.test_labels,
                    hold_prob: 1.0
                })
                print(f'=> step {i}, accuracy = {acc_value}')
