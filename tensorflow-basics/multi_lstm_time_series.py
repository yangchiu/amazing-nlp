import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

save_dir = 'multi_lstm_time_series'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class MilkProductionData():

    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.scaler = None
        self.get_data()

    def get_data(self):
        df = pd.read_csv(f'{save_dir}/monthly-milk-production.csv', index_col='Month')
        df.index = pd.to_datetime(df.index)
        df.plot(title='training data', ax=axes[0])
        self.train_set = df.head(156)
        self.test_set = df.tail(12)
        self.scaler = MinMaxScaler()
        self.train_set = self.scaler.fit_transform(self.train_set)

    def next_batch(self, batch_size, steps):
        rand_start = np.random.randint(0, len(self.train_set) - steps - 1)
        # train_set is a pandas dataframe with 1 column, 156 rows, so shape = [156, 1]
        y_batch = np.array(self.train_set[rand_start:rand_start+steps+1])
        # y_batch shape = [steps + 1, 1]
        return y_batch[:-1, :].reshape(-1, steps, 1), y_batch[1:, :].reshape(-1, steps, 1)
        # the data to be fed into rnn should be of shape [batch_size, steps, input_size]


if __name__ == '__main__':
    milk_production = MilkProductionData()

    # constants

    # input dimension
    num_inputs = 1
    # num of steps in each batch
    # rnn cell would be unrolled to this size
    steps = 12
    # num of hidden neurons in one rnn layer
    num_neurons = 100
    # output dimension
    num_outputs = 1

    learning_rate = 0.03
    epochs = 4000
    batch_size = 1

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, steps, num_inputs])
    y = tf.placeholder(tf.float32, shape=[None, steps, num_outputs])

    # layers
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # loss
    loss = tf.reduce_mean(tf.square(outputs - y))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)

        for i in range(epochs):

            x_batch, y_batch = milk_production.next_batch(batch_size, steps)
            sess.run(train, feed_dict={
                x: x_batch,
                y: y_batch
            })

            if i % 100 == 0:
                mse = loss.eval(feed_dict={
                    x: x_batch,
                    y: y_batch
                })
                print(f'=> epoch {i}, mse = {mse}')

        saver.save(sess, f'{save_dir}/')

    # restore session
    with tf.Session() as sess:

        saver.restore(sess, f'{save_dir}/')

        # repeatedly use last 12 month data to predict the 13rd month data
        train_seed = list(milk_production.train_set[-12:])
        for i in range(12):
            # the t+1 value predicted by model would be
            # append to train_seed
            # so use train_seed[-steps:] to acquire last 12 steps data
            x_batch = np.array(train_seed[-steps:]).reshape(-1, steps, 1)
            y_pred = sess.run(outputs, feed_dict={x: x_batch})
            train_seed.append(y_pred[0, -1, 0])

        results = milk_production.scaler.inverse_transform(
            np.array(train_seed[12:]).reshape(12, 1)
        )

        milk_production.test_set['Generated'] = results
        print(milk_production.test_set)
        milk_production.test_set.plot(title='prediction', ax=axes[1])
        plt.savefig(f'{save_dir}/fig.png')
