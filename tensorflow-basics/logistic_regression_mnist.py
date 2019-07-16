import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

save_dir = 'logistic_regression_mnist/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = 'MNIST_data'

if __name__ == '__main__':

    print(f'* read mnist data')
    mnist = input_data.read_data_sets(os.path.join(save_dir, data_dir), one_hot=True)
    print(f'=> shape of training data: {mnist.train.images.shape}')
    print(f'=> shape of test data: {mnist.test.images.shape}')
    print(f'=> shape of validation data: {mnist.validation.images.shape}')
    print(f'=> max value in image: {mnist.train.images[1].max()}')
    print(f'=> min value in image: {mnist.train.images[1].min()}')

    plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray')
    plt.savefig(os.path.join(save_dir, 'sample.png'))

    print(f'* create model and start training')
    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    # variables
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # graph

    # difference between matmul, multiply and *
    # matmul: multiplies matrix a by matrix b, e.g. [2, 3] matmul [3, 2] = [2, 2]
    # multiply: element-wise multiplication, e.g. [2, 3] multiply [2, 3] = [2, 3]
    # *: the same as multiply
    y = tf.matmul(x, W) + b

    # loss and optimizer

    # difference between sparse_softmax_cross_entropy_with_logits and softmax_cross_entropy_with_logits:
    # Having two different functions is a convenience, as they produce the same result.
    #
    # The difference is simple:
    #
    # For sparse_softmax_cross_entropy_with_logits,
    # labels must have the shape [batch_size] and the dtype int32 or int64.
    # Each label is an int in range [0, num_classes-1].
    #
    # For softmax_cross_entropy_with_logits,
    # labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
    # Labels used in softmax_cross_entropy_with_logits are the one hot version
    # of labels used in sparse_softmax_cross_entropy_with_logits.
    #
    # Another tiny difference is that with sparse_softmax_cross_entropy_with_logits,
    # you can give -1 as a label to have loss 0 on this label.
    #
    # the logits fed into softmax_cross_entropy_with_logits should be unscaled,
    # since it performs a softmax on logits internally for efficiency.
    # Do not call this softmax_cross_entropy_with_logits with the output of softmax,
    # as it will produce incorrect results.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        epochs = 1000
        batch_size = 100

        for iter in range(epochs):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

        matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        acc_output = sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels})

        print(f'=> accuracy = {acc_output}')



