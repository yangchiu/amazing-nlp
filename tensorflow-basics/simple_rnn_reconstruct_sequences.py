import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

save_dir = 'simple_rnn_reconstruct_sequences'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class TimeSeriesData():

    def __init__(self, num_points, tmin, tmax):
        print(f'* calling {TimeSeriesData.__init__.__name__}')

        self.num_points = num_points
        self.tmin = tmin
        self.tmax = tmax
        self.resolution = (tmax - tmin) / num_points
        self.t_data = np.linspace(tmin, tmax, num_points)
        self.y_true = np.sin(self.t_data)

    def generate_y_true(self, t_series):

        return np.sin(t_series)

    def next_batch(self, batch_size, steps, return_t_series=False):

        # grab a random starting point for each batch
        # numpy.random.rand(d0, d1, ..., dn)
        # create random samples from a uniform distribution [0, 1)
        # with shape [d0, d1, ..., dn]
        rand_start = np.random.rand(batch_size, 1) # shape [batch_size, 1]

        # convert to be on time series
        time_series_start = rand_start * (self.tmax - self.tmin - (steps * self.resolution))

        # create batch time series on x axis
        batch_t_series = time_series_start + np.arange(0, steps + 1) * self.resolution

        # create y data for time series in the batches
        batch_y_series = np.sin(batch_t_series)

        # return (t data, t+1 data)
        # trying to use t data to predict t+1 data
        if return_t_series:
            return batch_y_series[:, :-1].reshape(-1, steps, 1), \
                   batch_y_series[:, 1:].reshape(-1, steps, 1), \
                   batch_t_series
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
    plt.plot(time_series.t_data, time_series.y_true)
    plt.title('generated time series')
    plt.savefig(f'{save_dir}/fig.png')

    steps = 30
    y1, y2, ts = time_series.next_batch(1, steps, True)
    plt.subplot(2, 3, 2)
    plt.plot(ts.flatten()[1:], y2.flatten(), '*')
    plt.title('sampled time series')
    plt.savefig(f'{save_dir}/fig.png')

    plt.subplot(2, 3, 3)
    plt.plot(time_series.t_data, time_series.y_true, label='sin(t)')
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

    learning_rate = 0.0001
    epochs = 2000
    batch_size = 1

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, steps, num_inputs])
    y = tf.placeholder(tf.float32, shape=[None, steps, num_outputs])

    # layers
    hidden_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons)
    # using an OutputProjectionWrapper is the simplest solution to
    # reduce the dimensionality of the RNN’s output sequences down to
    # just one value per time step.
    #
    # from [batch_size, steps, hidden_layer_size]
    # to [batch_size, steps, output_layer_size]
    output_projection_cell = tf.contrib.rnn.OutputProjectionWrapper(
        hidden_rnn_cell,
        output_size=num_outputs
    )

    # static_rnn vs dynamic_rnn:
    # tf.nn.rnn creates an unrolled graph for a fixed RNN length.
    # That means, if you call tf.nn.rnn with inputs having 200 time steps
    # you are creating a static graph with 200 RNN steps.
    # First, graph creation is slow.
    # Second, you’re unable to pass in longer sequences (> 200) than you’ve originally specified.
    # tf.nn.dynamic_rnn solves this.
    # It uses a tf.While loop to dynamically construct the graph when it is executed.
    # That means graph creation is faster and you can feed batches of variable size.

    # The tf.nn.dynamic_rnn() or tf.nn.rnn() operations allow to specify the initial state
    # of the RNN using the initial_state parameter.
    # If you don't specify this parameter,
    # the hidden states will be initialized to zero vectors at the beginning of
    # each training batch.

    # outputs: The RNN output Tensor
    # shape = [batch_size, steps, cell_state_size (number of units in the cell)]
    #
    # states: The final state
    # shape = [batch_size, cell_state_size (number of units in the cell)]
    outputs, states = tf.nn.dynamic_rnn(output_projection_cell, x, dtype=tf.float32)

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

        # only for evaluating the shapes of outputs and states
        x_batch, y_batch = time_series.next_batch(batch_size, steps)
        hidden_outputs, hidden_states = tf.nn.dynamic_rnn(hidden_rnn_cell, x, dtype=tf.float32)

        hidden_outputs_eval = hidden_outputs.eval(feed_dict={
            x: x_batch
        })
        hidden_states_eval = hidden_states.eval(feed_dict={
            x: x_batch
        })
        print(f'=> shape of rnn cell outputs is {hidden_outputs_eval.shape}')
        print(f'=> shape of rnn cell states is {hidden_states_eval.shape}')

        outputs_eval = outputs.eval(feed_dict={
            x: x_batch
        })
        states_eval = states.eval(feed_dict={
            x: x_batch
        })
        print(f'=> shape of projection cell outputs is {outputs_eval.shape}')
        print(f'=> shape of projection cell states is {states_eval.shape}')

        for i in range(epochs):

            # x_batch is 0 ~ t-1 data
            # y_batch is 1 ~ t data
            #
            # when 0 ~ t-1 data fed into model
            # the outputs is 1 ~ t data
            #
            # compare the outputs with y_batch
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

        saver.save(sess, f'{save_dir}/')

    # predict a time series t+1 based on trained model
    with tf.Session() as sess:
        saver.restore(sess, f'{save_dir}/')

        # generate new time series
        t_series = np.linspace(30, 30 + time_series.resolution * (steps + 1), steps + 1)
        x_new = np.sin(np.array(t_series[:-1].reshape(-1, steps, num_inputs)))

        # t_series is [0, 31]
        # x_new is [0, 30]
        # y_pred is [1, 31]
        # the first value of y_pred doesn't have previous state can be referenced
        # so the prediction is incorrect

        y_pred = sess.run(outputs, feed_dict={x: x_new})

        plt.subplot(2, 3, 4)
        plt.plot(t_series[:-1], time_series.generate_y_true(t_series[:-1]),
                 'bo', markersize=15, alpha=0.5, label='Training Instance')
        plt.plot(t_series[1:], time_series.generate_y_true(t_series[1:]),
                 'ko', markersize=10, label='target')
        plt.plot(t_series[1:], y_pred[0,:,0], 'r.', markersize=10, label='prediction')
        plt.xlabel('t')
        plt.title('testing model')
        plt.legend()

    # generate new sequences
    with tf.Session() as sess:
        saver.restore(sess, f'{save_dir}/')

        zero_seeds = [0 for i in range(steps)]
        for i in range(len(time_series.t_data) - steps):
            # the t+1 value predicted by model would be
            # append to zero_seeds
            # so use zero_seeds[-steps:] to acquire last # of steps data
            x_batch = np.array(zero_seeds[-steps:]).reshape(-1, steps, num_inputs)
            y_pred = sess.run(outputs, feed_dict={x: x_batch})
            zero_seeds.append(y_pred[0, -1, 0])

        plt.subplot(2, 3, 5)
        plt.plot(time_series.t_data, zero_seeds, "b-")
        plt.plot(time_series.t_data[:steps], zero_seeds[:steps], "r", linewidth=3)
        plt.xlabel("t")
        plt.title('generate new sequences')
        plt.savefig(f'{save_dir}/fig.png')
