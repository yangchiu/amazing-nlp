import numpy as np
import os
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Bidirectional, MaxPool1D, GlobalMaxPooling1D, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.100d.txt'

save_dir = 'lstm_comments_classification/'
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
                  weights=[toxic.embedding],
                  trainable=False)(inputs)
    # output shape = (120, output_dim=100)
    print(x)

    # input shape = (120, 100)
    x = Bidirectional(LSTM(units=15, return_sequences=True))(x)
    # output shape = (120, units * 2 = 30)
    #                             ^
    #                             |____ Bidirectional causes the output dim * 2

    # input shape = (120, 30)
    x = GlobalMaxPooling1D()(x)
    # input shape = (30)

    outputs = Dense(units=len(toxic.possible_labels),
                    activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])

    model.fit(x=toxic.indexed_comments,
              y=toxic.targets,
              batch_size=128,
              epochs=3,
              validation_split=0.3)
