import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

save_dir = 'autoencoder_pca_30to2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename = 'anonymized_data.csv'


def get_data():
    print(f'* calling {get_data.__name__}')

    df = pd.read_csv(os.path.join(save_dir, filename))
    print(f'=> get dataframe with shape {df.shape}')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop('Label', axis=1))
    labels = df['Label']
    print(f'=> inputs shape {scaled_data.shape}')
    print(f'=> labels shape {labels.shape}')
    return scaled_data, labels


if __name__ == '__main__':

    inputs, labels = get_data()

    # constants

    num_inputs = 30
    num_hidden = 2
    num_outputs = num_inputs

    learning_rate = 0.01

    # placeholders

    x = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # layers

    hidden = tf.contrib.layers.fully_connected(x, num_hidden, activation_fn=None)
    outputs = tf.contrib.layers.fully_connected(hidden, num_outputs, activation_fn=None)

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

            sess.run(train, feed_dict={x: inputs})

        lowered_dimension_outputs = hidden.eval(feed_dict={x: inputs})

    print(f'=> inputs shape = {inputs.shape}')
    print(f'=> lowered dimension outputs shape = {lowered_dimension_outputs.shape}')

    plt.scatter(lowered_dimension_outputs[:, 0], lowered_dimension_outputs[:, 1], c=labels)
    plt.savefig(os.path.join(save_dir, 'fig.png'))
