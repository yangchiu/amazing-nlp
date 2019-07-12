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
    num_hidden = 100
    # output dimension
    num_outputs = 1
    # num of rnn layers
    num_layers = 3

    learning_rate = 0.005
    epochs = 4000
    batch_size = 1

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, steps, num_inputs])
    y = tf.placeholder(tf.float32, shape=[None, steps, num_outputs])
    # If we consider an individual LSTM cell, for each training sample it processes it has two other inputs:
    # the previous output from the cell h(t-1)
    # and the previous state variable s(t-1)
    #
    # These two inputs, h and s, are what is required to load the full state data into an LSTM cell.
    # Remember also that h and s for each sample are actually vectors with the size equal to the hidden layer size.
    # Therefore, for all the samples in the batch,
    # for a single LSTM cell we have state data required of shape (2, batch_size, hidden_size).
    #
    # Finally, if we have stacked LSTM cell layers, we need state variables for each layer – num_layers.
    # This gives the final shape of the state variables: (num_layers, 2, batch_size, num_hidden).
    init_state = tf.placeholder(tf.float32, shape=[num_layers, 2, batch_size, num_hidden])

    keep_prob = tf.placeholder(tf.float32)

    # The tf.unstack command creates a number of tensors, each of shape (2, batch_size, num_hidden),
    # from the init_state tensor, one for each stacked LSTM layer (num_layers).
    #
    # These tensors are then loaded into a specific TensorFlow data structure:
    # LSTMStateTuple, which is the required for input into the LSTM cells.
    state_per_layer_list = tf.unstack(init_state, axis=0)
    lstm_tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(
            state_per_layer_list[idx][0],
            state_per_layer_list[idx][1])
            for idx in range(num_layers)
        ]
    )

    # layers

    # do not use [cell] * num_layers to create multiple rnn cells
    # do not use [cell for _ range(num_layers)] to create multiple rnn cells
    # else you will encounter "ValueError: Dimensions must be equal" error
    # use the code below to create "separate" rnn cell instances,
    # instead of different pointers point to the same rnn cell instance.
    cells = []
    for i in range(num_layers):
        # set the forget bias values to be equal to 1.0,
        # which helps guard against repeated low forget gate outputs causing vanishing gradients
        cells.append(tf.contrib.rnn.LSTMCell(num_units=num_hidden, forget_bias=1.0, state_is_tuple=True))
    for i in range(num_layers):
        cells[i] = tf.contrib.rnn.DropoutWrapper(cells[i], output_keep_prob=keep_prob)

    multicell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    outputs, states = tf.nn.dynamic_rnn(multicell, x, dtype=tf.float32, initial_state=lstm_tuple_state)
    # reshape outputs from [batch_size, steps, num_hidden]
    # to [batch_size * steps, num_hidden]
    # so we can perform x * W + b
    # to reduce dimension from num_hidden to num_outputs
    outputs = tf.reshape(outputs, [-1, num_hidden])

    logits_w = init_weights([num_hidden, num_outputs])
    logits_b = init_bias([num_outputs])
    logits = tf.nn.xw_plus_b(outputs, logits_w, logits_b)

    # reshape logits from [batch_size * steps, num_outputs]
    # to [batch_size, steps, num_outputs]
    # so it can be compared with y_true
    logits = tf.reshape(logits, [batch_size, steps, num_outputs])

    # loss
    loss = tf.reduce_mean(tf.square(logits - y))

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
            current_state = np.zeros((num_layers, 2, batch_size, num_hidden))
            _, loss_ = sess.run([train, loss], feed_dict={
                x: x_batch,
                y: y_batch,
                init_state: current_state,
                keep_prob: 0.5
            })

            if i % 100 == 0:
                mse = loss.eval(feed_dict={
                    x: x_batch,
                    y: y_batch,
                    init_state: current_state,
                    keep_prob: 1.0
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
            y_pred = sess.run(logits, feed_dict={
                x: x_batch,
                init_state: current_state,
                keep_prob: 1.0
            })
            train_seed.append(y_pred[0, -1, 0])

        results = milk_production.scaler.inverse_transform(
            np.array(train_seed[12:]).reshape(12, 1)
        )

        milk_production.test_set['Generated'] = results
        print(milk_production.test_set)
        milk_production.test_set.plot(title='prediction', ax=axes[1])
        plt.savefig(f'{save_dir}/fig.png')
