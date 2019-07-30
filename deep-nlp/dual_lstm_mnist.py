import os
import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Bidirectional, GlobalMaxPooling1D, Dense
from keras.layers import Lambda, Concatenate
from keras.models import Model
import keras.backend as K

save_dir = 'dual_lstm_mnist/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mnist_filename = 'mnist.csv'


def get_mnist():
    print(f'* calling {get_mnist.__name__}')

    df = pd.read_csv(os.path.join(save_dir, mnist_filename))
    data = df.values

    # numpy.random.shuffle(x):
    # modify a sequence "in-place" by shuffling its contents.
    #
    # this function only shuffles the array along the first axis of a multi-dimensional array.
    # the order of sub-arrays is changed but their contents remains the same.
    np.random.shuffle(data)

    print(f'=> data shape = {data.shape}')
    # data shape = (42000, 785)
    # the 1st column is label
    # rest of columns are 28 * 28 = 784 pixels

    # treat image data as (steps, input_dim)
    # just like a time-series
    x = data[:, 1:].reshape(-1, 28, 28) / 255
    print(f'=> x shape = {x.shape}')

    y = data[:, 0]
    print(f'=> y shape = {y.shape}')

    return x, y


if __name__ == '__main__':

    x_data, y_data = get_mnist()

    # config

    input_dim = steps = 28
    hidden_dim = 15

    # model

    inputs = Input(shape=(steps, input_dim))

    # up-down
    #
    #      input_dim
    #  s ==============>
    #  t ==============>
    #  e ==============>
    #  p ==============>
    #  s ==============>
    #
    # input shape = (28, 28)
    x1 = Bidirectional(LSTM(units=hidden_dim,
                            return_sequences=True))(inputs)
    # output shape = (28, 15 * 2)

    # input shape = (28, 15 * 2)
    # steps dimension would be reduced
    x1 = GlobalMaxPooling1D()(x1)
    # output shape = (, 15 * 2)

    # left-right
    #
    #       input_dim
    #  s | | | | | | | |
    #  t | | | | | | | |
    #  e | | | | | | | |
    #  p | | | | | | | |
    #  s | | | | | | | |
    #    v v v v v v v v
    #
    #  => transpose the input data to achieve this

    # Lambda: wraps arbitrary expressions as a Layer object.
    # permute_dimensions: just act like np.transpose
    x2 = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))(inputs)

    # input shape = (28, 28)
    x2 = Bidirectional(LSTM(units=hidden_dim,
                            return_sequences=True))(x2)
    # output shape = (28, 15 * 2)

    # input shape = (28, 15 * 2)
    x2 = GlobalMaxPooling1D()(x2)
    # output shape = (, 15 * 2)

    # input shape = (, 30) + (, 30)
    x_all = Concatenate(axis=1)([x1, x2])
    # output shape = (, 60)

    # 0 ~ 9 digits, so units = 10
    outputs = Dense(units=10, activation='softmax')(x_all)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # could be slow
    model.fit(x=x_data,
              y=y_data,
              batch_size=32,
              epochs=10,
              validation_split=0.3)
