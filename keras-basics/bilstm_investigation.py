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

    # test LSTM

    x = LSTM(units=hidden_dim,
             # return_state:
             # Whether to return the last state in addition to the output.
             # The returned elements of the states list are:
             # "the hidden state" and "the cell state", respectively.
             return_state=True,
             # return_sequences:
             # Whether to return the last output in the output sequence,
             # or the full sequence.
             return_sequences=True)(inputs)

    model = Model(inputs=inputs, outputs=x)
    # if return_state = True,
    # LSTM would return 3 arrays,
    # the first is the output,
    # the second is the last hidden state,
    # the third is the last cell state
    #
    # hidden state = the last output
    output, hidden, cell = model.predict(random_data)
    print(f'* LSTM Test, return_state=True, return_sequences=True')
    print(f'output shape = {output.shape}')
    print(output)
    print(f'hidden shape = {hidden.shape}')
    print(hidden)
    print(f'cell shape = {cell.shape}')
    print(cell)

    print('------')

    # test BiLSTM

    x = Bidirectional(LSTM(units=hidden_dim,
                           return_state=True,
                           return_sequences=True))(inputs)

    model = Model(inputs=inputs, outputs=x)
    # if return_state = True,
    # BiLSTM would return 5 arrays,
    # the first is the output,
    # the second is the last hidden state (t=T) of forward direction,
    # the third is the last cell state (t=T) of forward direction,
    # the forth is the last hidden state (t=1) of backward direction,
    # the fifth is the last cell state (t=1) of backward direction
    #
    # for forward direction, hidden state = the last output
    # for backward direction, hidden state = the first output
    #
    # both forward and backward LSTM would return an output with dimension [hidden_dim]
    # biLSTM would concat these 2 output
    # so the output dimension of biLSTM would be [2 * hidden_dim]
    # the output would be like: [ forward_output backward_output ]
    output, hidden_f, cell_f, hidden_b, cell_b = model.predict(random_data)
    print(f'* BiLSTM Test, return_state=True, return_sequences=True')
    print(f'output shape = {output.shape}')
    print(output)
    print(f'forward hidden shape = {hidden_f.shape}')
    print(hidden_f)
    print(f'forward cell shape = {cell_f.shape}')
    print(cell_f)
    print(f'backward hidden shape = {hidden_b.shape}')
    print(hidden_b)
    print(f'backward cell shape = {cell_b.shape}')
    print(cell_b)
