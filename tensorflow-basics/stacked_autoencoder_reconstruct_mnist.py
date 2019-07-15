import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

save_dir = 'stacked_autoencoder_reconstruct_mnist/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
data_dir = 'MNIST_data/'


if __name__ == '__main__':

    mnist = input_data.read_data_sets(os.path.join(save_dir, data_dir), one_hot=True)
    print(f'=> get {mnist.train.num_examples} training data')

    # constants

    num_inputs = 784

    num_hidden1 = 392
    num_hidden2 = 196
    num_hidden3 = num_hidden1
    num_outputs = num_inputs

    learning_rate = 0.01

    # placeholders

    x = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # init weights
    # When initializing a deep network,
    # it is in principle advantageous to keep the scale of the input variance constant,
    # so it does not explode or diminish by reaching the final layer.
    # This initializer use the following formula:
    #
    #   if mode='FAN_IN': # Count only number of input connections.
    #     n = fan_in
    #   elif mode='FAN_OUT': # Count only number of output connections.
    #     n = fan_out
    #   elif mode='FAN_AVG': # Average number of inputs and output connections.
    #     n = (fan_in + fan_out)/2.0
    #
    #   truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
    weights_initializer = tf.variance_scaling_initializer()

    w1 = tf.Variable(weights_initializer([num_inputs, num_hidden1]), dtype=tf.float32)
    w2 = tf.Variable(weights_initializer([num_hidden1, num_hidden2]), dtype=tf.float32)
    w3 = tf.Variable(weights_initializer([num_hidden2, num_hidden3]), dtype=tf.float32)
    w4 = tf.Variable(weights_initializer([num_hidden3, num_outputs]), dtype=tf.float32)

    b1 = tf.Variable(tf.zeros(num_hidden1))
    b2 = tf.Variable(tf.zeros(num_hidden2))
    b3 = tf.Variable(tf.zeros(num_hidden3))
    b4 = tf.Variable(tf.zeros(num_outputs))

    # layers

    # [batch_size, num_inputs] x [num_inputs, num_hidden1] = [batch_size, num_hidden1]
    hidden_layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # [batch_size, num_hidden1] x [num_hidden1, num_hidden2] = [batch_size, num_hidden2]
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w2) + b2)
    # [batch_size, num_hidden2] x [num_hidden2, num_hidden3] = [batch_size, num_hidden3]
    hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w3) + b3)
    # [batch_size, num_hidden3] x [num_hidden3, num_outputs] = [batch_size, num_outputs]
    output_layer = tf.matmul(hidden_layer3, w4) + b4

    # loss
    loss = tf.reduce_mean(tf.square(output_layer - x)) # mse

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)

        epochs = 5
        batch_size = 150

        for i in range(epochs):

            num_batches = mnist.train.num_examples // batch_size

            for j in range(num_batches):

                x_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(train, feed_dict={x: x_batch})

            training_loss = loss.eval(feed_dict={x: x_batch})

            print(f'=> epoch {i}, training loss = {training_loss}')

        saver.save(sess, save_dir)

    with tf.Session() as sess:

        saver.restore(sess, save_dir)

        num_test_images = 10
        results = output_layer.eval(feed_dict={x: mnist.test.images[:10]})

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
        for i in range(num_test_images):
            axes[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            axes[1][i].imshow(np.reshape(results[i], (28, 28)))

        plt.savefig(os.path.join(save_dir, 'fig.png'))
