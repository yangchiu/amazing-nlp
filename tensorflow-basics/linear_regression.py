import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

save_dir = 'linear_regression'


def create_data():
    print(f'* calling {create_data.__name__}')

    # create x data, 1 million points from 0 to 10
    x_data = np.linspace(0, 10, 1000000)

    # create normal(0, 1) noise
    noise = np.random.randn(len(x_data))

    # create y data
    # y = 0.5x + 5 + noise
    y_true = 0.5 * x_data + 5 + noise
    print(f'=> create data with m = 0.5, b = 5')

    df_data = pd.concat(
        [
            pd.DataFrame(data=x_data, columns=['X Data']),
            pd.DataFrame(data=y_true, columns=['Y']),
        ],
        axis=1
    )

    print(df_data.head())

    return x_data, y_true, df_data


if __name__ == '__main__':

    x_data, y_true, df_data = create_data()

    batch_size = 10
    learning_rate = 0.001

    print(f'=> create model and start training')

    # variables
    m = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())

    # placeholders
    x_ph = tf.placeholder(tf.float32, shape=[batch_size])
    y_ph = tf.placeholder(tf.float32, shape=[batch_size])

    # graph
    y_model = m * x_ph + b

    # loss function
    loss = tf.reduce_sum(tf.square(y_ph - y_model))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # init variables
    init = tf.global_variables_initializer()

    # run session
    with tf.Session() as sess:

        sess.run(init)

        epochs = 1000

        for i in range(epochs):

            # numpy.random.randint(low, high=None, size=None, dtype='l')
            # Return random integers from the “discrete uniform” distribution in the interval [low, high).
            # If high is None, results are from [0, low).
            random_indexes = np.random.randint(len(x_data), size=batch_size)

            feed = {
                x_ph: x_data[random_indexes],
                y_ph: y_true[random_indexes]
            }

            sess.run(train, feed_dict=feed)

        model_m, model_b = sess.run([m, b])
        print(f'=> get model_m = {model_m}, model_b = {model_b}')

    y_hat = model_m * x_data + model_b

    df_data.sample(n=250).plot(kind='scatter',
                               x='X Data',
                               y='Y',
                               title='data scatter plot',
                               ax=axes[0])
    fig = df_data.sample(n=250).plot(kind='scatter',
                                     x='X Data',
                                     y='Y',
                                     title='data scatter plot + model prediction',
                                     ax=axes[1])
    fig.plot(x_data, y_hat, 'r')
    plt.savefig(f'{save_dir}/fig.png')

