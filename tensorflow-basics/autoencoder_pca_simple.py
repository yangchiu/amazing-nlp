import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

save_dir = 'autoencoder_pca_simple/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def create_random_data():
    print(f'* calling {create_random_data.__name__}')

    data = make_blobs(n_samples=100, n_features=3, centers=2)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[0])
    data_x = scaled_data[:, 0]
    data_y = scaled_data[:, 1]
    data_z = scaled_data[:, 2]

    plt.figure(figsize=(10, 5))
    plt.subplot(121, projection='3d')
    plt.scatter(data_x, data_y, data_z, c=data[1], cmap=plt.cm.get_cmap('RdYlBu'))
    #plt.show()
    plt.savefig(os.path.join(save_dir, 'fig.png'))

    return scaled_data, data[1]


if __name__ == '__main__':

    data, labels = create_random_data()

    # constants

    num_inputs = 3
    # hidden layer would be the lower dimension representation of inputs
    num_hidden = 2
    # for an autoencoder, num_outputs always equal to num_inputs
    # the outputs of an autoencoder would recontruct the inputs
    num_outputs = num_inputs

    learning_rate = 0.01

    # placeholders

    x = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # layers

    hidden = tf.contrib.layers.fully_connected(x, num_hidden, activation_fn=None)
    outputs = tf.contrib.layers.fully_connected(hidden, num_outputs, activation_fn=None)

    # w_ih = 3 x 2
    # w_ho = 2 x 3

    # loss
    loss = tf.reduce_mean(tf.square(outputs - x)) # mse

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:

        sess.run(init)

        epochs = 1000

        for i in range(epochs):

            sess.run(train, feed_dict={x: data})

        lowered_dim_outputs = hidden.eval(feed_dict={x: data})

    print(f'=> input shape = {data.shape}')
    print(f'=> lowered dimension outputs = {lowered_dim_outputs.shape}')

    plt.subplot(122)
    plt.scatter(lowered_dim_outputs[:, 0], lowered_dim_outputs[:, 1], c=labels, cmap=plt.cm.get_cmap('RdYlBu'))
    plt.savefig(os.path.join(save_dir, 'fig.png'))
