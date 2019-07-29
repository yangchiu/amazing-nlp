import numpy as np
import os
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Conv1D, MaxPool1D, GlobalMaxPooling1D, Dense
from keras.models import Model

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.100d.txt'

save_dir = 'cnn_comments_classification/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_filename = 'train.csv'


class Glove:

    def __init__(self, filepath):
        print(f'* calling Glove()')

        self.embedding = []
        self.word2idx = {}
        self.idx2word = []
        with open(filepath, encoding='utf-8') as f:
            # it's just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for i, line in enumerate(f):

                values = line.split()

                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)

                self.embedding.append(vector)
                self.idx2word.append(word)
                self.word2idx[word] = i

            print(f'=> found {len(self.idx2word)} word vectors')

        self.embedding = np.array(self.embedding)
        self.vocab_size, self.embedding_size = self.embedding.shape


class ToxicComments:

    def __init__(self, pretrained_word_embedding):
        print(f'* calling {ToxicComments.__init__.__name__}')

        self.pretrained = pretrained_word_embedding

        self.comments = None
        self.targets = None
        self.possible_labels = None
        self.load_data()

        self.tokenizer = None
        self.vocab_size = 50000
        self.max_comment_length = 120
        self.indexed_comments = None
        self.tokenize()

        self.embedding = None
        self.build_word_embedding()

    def load_data(self):
        print(f'* calling {ToxicComments.load_data.__name__}')

        train = pd.read_csv(os.path.join(save_dir, data_filename))

        self.possible_labels = train.columns.values[2:]
        print(f'=> possible labels are:')
        print(self.possible_labels)

        # one comment could have one or more labels
        self.targets = train[self.possible_labels].values

        self.comments = train['comment_text'].values

        # replace \xa0 with white-space
        for i, line in enumerate(self.comments):
            self.comments[i] = MultiReplacer.replace(self.comments[i])

        # remove all punctuations
        for i, line in enumerate(self.comments):
            self.comments[i] = re.sub(r'[^a-zA-Z\d\s]', '', self.comments[i])

        # remove empty lines
        for i, line in reversed(list(enumerate(self.comments))):
            if len(self.comments[i]) == 0:
                self.comments = np.delete(self.comments, i)
                self.targets = np.delete(self.targets, i, axis=0)

        print(f'=> get {len(self.comments)} comments')
        print(f'=> get {len(self.targets)} targets')

    def tokenize(self):
        print(f'* calling {ToxicComments.tokenize.__name__}')

        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(self.comments)
        self.indexed_comments = self.tokenizer.texts_to_sequences(self.comments)

        # remove empty lines
        for i, line in reversed(list(enumerate(self.indexed_comments))):
            if len(self.indexed_comments[i]) == 0:
                del self.indexed_comments[i]
                self.comments = np.delete(self.comments, i)
                self.targets = np.delete(self.targets, i, axis=0)

        print(f'=> get {len(self.indexed_comments)} indexed_comments')
        print(f'=> get {len(self.comments)} comments')
        print(f'=> get {len(self.targets)} targets')

        print(f'=> max comment length = {max(len(s) for s in self.indexed_comments)}')
        print(f'=> min comment length = {min(len(s) for s in self.indexed_comments)}')

        self.indexed_comments = pad_sequences(self.indexed_comments, maxlen=self.max_comment_length)
        print(f'=> pad indexed_comments to shape {self.indexed_comments.shape}')

    def build_word_embedding(self):
        print(f'* calling {ToxicComments.build_word_embedding.__name__}')

        # using pretrained word embedding to build
        # word embedding for words in comments data

        # keras tokenizer reserve index 0 for padding
        num_words = self.vocab_size + 1

        self.embedding = np.zeros((num_words, self.pretrained.embedding_size))

        # word vector for <pad> is all zeros
        for word, i in self.tokenizer.word_index.items():
            if i < self.vocab_size:
                index = self.pretrained.word2idx.get(word)
                # words not found in pretrained will be all zeros
                if index is not None:
                    self.embedding[i] = self.pretrained.embedding[index]

        print(f'=> get word embedding with shape = {self.embedding.shape}')


class MultiReplacer:
    dict = {
        # &nbsp
        '\xa0': ' ',
        # dash
        '\u2011': '\u002D',
        '\u2212': '\u002D',
        '\u2013': '\u002D',
        '\u2014': '\u002D',
        # single quotation
        '\u2018': '\u0027',
        '\u2019': '\u0027',
        # left double quotation
        '\u201C': '\u0022',
        # right double quotation
        '\u201D': '\u0022'
    }

    regex = re.compile(f'{"|".join(map(re.escape, dict.keys()))}')

    @classmethod
    def replace(cls, text):
        return cls.regex.sub(
            lambda mo: cls.dict[mo.string[mo.start():mo.end()]],
            text
        )


if __name__ == '__main__':

    glove = Glove(glove_filepath)
    toxic = ToxicComments(glove)

    # inputs

    inputs = Input(shape=(toxic.max_comment_length,))

    # model

    # input shape = (seq_len=120, 1)
    x = Embedding(input_dim=toxic.vocab_size + 1,
                  input_length=toxic.max_comment_length,
                  output_dim=glove.embedding_size,
                  # load pretrained word embeddings into the Embedding layer
                  weights=[toxic.embedding],
                  trainable=False)(inputs)
    # output shape = (120, output_dim=100)

    #
    # use CNN on time sequence data here
    # to show not only RNN, CNN can be used to predict time-series, too
    # the convolution and max pooling would be performed along time steps axis
    #

    # Conv1D creates a convolution kernel that is convolved
    # with the layer input over a single spatial (or "temporal") dimension
    # filters = output dimension,
    # kernel_size = the length of the 1D convolution window
    # strides default to 1

    # input shape = (120, 100)
    x = Conv1D(filters=128,
               kernel_size=3,
               activation='relu')(x)
    # output_dim = ((input_dim - window_size + 2 * padding) / strides) + 1
    #            = ((   120    -      3      + 2 *    0  ) /     1   ) + 1
    #            = 117 + 1
    #            = 118
    # output shape = (118, filters = 128)

    # MaxPooling1D: Max pooling operation for "temporal" data.
    # pool_size = size of the max pooling window,
    # if strides = None, it will default to pool_size

    # input shape = (118, 128)
    x = MaxPool1D(pool_size=3)(x)
    # output_dim = ((input_dim - window_size + 2 * padding) / strides) + 1
    #            = ((   118    -      3      + 2 *    0   ) /    3   ) + 1
    #            = 115 / 3 + 1
    #            = 38 + 1
    #            = 39
    # output shape = (39, 128)

    # input shape = (39, 128)
    x = Conv1D(filters=128,
               kernel_size=3,
               activation='relu')(x)
    # output_dim = ((input_dim - window_size + 2 * padding) / strides) + 1
    #            = ((    39    -      3      + 2 *    0   ) /    1   ) + 1
    #            = 36 + 1
    #            = 37
    # output shape = (37, 128)

    # input shape = (37, 128)
    x = MaxPool1D(pool_size=3)(x)
    # output_dim = ((input_dim - window_size + 2 * padding) / strides) + 1
    #            = ((    37    -      3      + 2 *    0   ) /    3   ) + 1
    #            = 34 / 3 + 1
    #            = 11 + 1
    #            = 12
    # output shape = (12, 128)

    # input shape = (12, 128)
    x = Conv1D(filters=128,
               kernel_size=3,
               activation='relu')(x)
    # output_dim = ((input_dim - window_size + 2 * padding) / strides) + 1
    #            = ((    12    -      3      + 2 *    0   ) /    1   ) + 1
    #            = 9 / 1 + 1
    #            = 10
    # output shape = (10, 128)

    # GlobalMaxPooling1D for temporal data takes the max vector over the steps dimension.
    # So a tensor with shape [10, 4, 10] becomes a tensor with shape [10, 10] after global pooling.
    #
    # MaxPooling1D takes the max over the steps too but constrained to a pool_size for each stride.
    # So a [10, 4, 10] tensor with pooling_size=2 and stride=1 is a [10, 3, 10] tensor
    # after MaxPooling(pooling_size=2, stride=1)

    # input shape = (10, 128)
    x = GlobalMaxPooling1D()(x)
    # output shape = (128)

    # input shape = (128)
    x = Dense(units=128,
              activation='relu')(x)
    # output shape = (units=128)

    # input shape = (128)
    outputs = Dense(units=len(toxic.possible_labels),
                    # use sigmoid here
                    # because sigmoid can be used in multi-labels classification
                    # for softmax, each class is mutual exclusive
                    activation='sigmoid')(x)
    # output shape = (units=6)

    model = Model(inputs, outputs)
    # if it's a multi-labels classification task,
    # use binary_crossentropy
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x=toxic.indexed_comments,
              y=toxic.targets,
              # if batch_size unspecified, will default to 32.
              batch_size=128,
              epochs=10,
              validation_split=0.3)
