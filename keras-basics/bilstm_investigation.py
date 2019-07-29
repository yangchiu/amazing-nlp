from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np


if __name__ == '__main__':

    steps = 8
    input_dim = 2
    hidden_dim = 3

    # randn(d0, d1, ..., dn)
    # return an array of shape (d0, d1, ..., dn)
    random_data = np.random.randn(1, steps, input_dim)

    inputs = Input(shape=(steps, input_dim))

    x = LSTM(units=hidden_dim,
             return_state=True,
             return_sequences=True)(inputs)

    model = Model(inputs=inputs, outputs=x)

    out = model.predict(random_data)
    print(out)