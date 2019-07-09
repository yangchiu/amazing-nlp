import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

save_dir = 'rnn_reconstruct_sequences'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class TimeSeriesData():

    def __init__(self, num_points, xmin, xmax):
        print(f'* calling {TimeSeriesData.__init__.__name__}')

        self.num_points = num_points
        self.xmin = xmin
        self.xmax = xmax
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def generate_y_true(self, x_series):

        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_x_series=False):

        # grab a random starting point for each batch
        # numpy.random.rand(d0, d1, ..., dn)
        # create random samples from a uniform distribution [0, 1)
        # with shape [d0, d1, ..., dn]
        rand_start = np.random.rand(batch_size, 1) # shape [batch_size, 1]

        # convert to be on time series
        time_series_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

        # create batch time series on x axis
        batch_x_series = time_series_start + np.arange(0, steps + 1) * self.resolution

        # create y data for time series in the batches
        batch_y_series = np.sin(batch_x_series)

        # return (t data, t+1 data)
        # trying to use t data to predict t+1 data
        if return_x_series:
            return batch_y_series[:, :-1].reshape(-1, steps, 1), \
                   batch_y_series[:, 1:].reshape(-1, steps, 1), \
                   batch_x_series
        else:
            return batch_y_series[:, :-1].reshape(-1, steps, 1), \
                   batch_y_series[:, 1:].reshape(-1, steps, 1)


def init_weights(shape):
    init_random_tensor = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_tensor)


def init_bias(shape):
    init_bias_tensor = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_tensor)


if __name__ == '__main__':

    print(f'======== generate fake time series ========')

    time_series = TimeSeriesData(250, 0, 10)

    plt.subplot(2, 3, 1)
    plt.plot(time_series.x_data, time_series.y_true)
    plt.title('generated time series')
    plt.savefig(f'{save_dir}/fig.png')

    steps = 30
    y1, y2, ts = time_series.next_batch(1, steps, True)
    plt.subplot(2, 3, 2)
    plt.plot(ts.flatten()[1:], y2.flatten(), '*')
    plt.title('sampled time series')
    plt.savefig(f'{save_dir}/fig.png')

    plt.subplot(2, 3, 3)
    plt.plot(time_series.x_data, time_series.y_true, label='sin(t)')
    plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='sampled time series')
    plt.legend()
    plt.title('generated & sampled time series overlapping')
    #plt.tight_layout()
    plt.savefig(f'{save_dir}/fig.png')

    # create model

    # constants
    # using y data to predict y+1 data
    # both input and output are 1-D
    num_inputs = num_outputs = 1

    # number of RNN cells
    num_neurons = 100

    # number of hidden RNN layers
    num_layers = 3

    learning_rate = 0.0001

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, steps, num_inputs])
    y = tf.placeholder(tf.float32, shape=[None, steps, num_outputs])

    # layers
    # use MultiRNNCell to stack multiple RNN cells
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        # the only required argument is num_units
        # num_units: The number of (hidden) units in the LSTM cell.
        tf.contrib.rnn.BasicRNNCell(num_units=num_neurons),
        output_size=num_outputs)

    # outputs: The RNN output Tensor
    # shape = [batch_size, steps, cell_state_size (number of hidden units in the cell)]
    #
    # states: The final state
    # shape = [batch_size, cell_state_size (number of hidden units in the cell)]
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    #logits_w = init_weights([num_neurons, num_outputs])
    #logits_b = init_bias([num_outputs])
    #logits = tf.matmul(outputs, logits_w) + logits_b
    #logits = tf.nn.xw_plus_b(outputs, logits_w, logits_b)

    # loss
    loss = tf.reduce_mean(tf.square(outputs - y))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:

        sess.run(init)

        epochs = 2000
        batch_size = 1

        for i in range(epochs):

            x_batch, y_batch = time_series.next_batch(batch_size, steps)
            sess.run(train, feed_dict={
                x: x_batch,
                y: y_batch
            })

            if i % 100 == 0:
                mse = loss.eval(feed_dict={
                    x: x_batch,
                    y: y_batch
                })
                print(f'=> step {i}, mse = {mse}')



