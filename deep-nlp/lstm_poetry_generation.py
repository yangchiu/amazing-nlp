import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.100d.txt'

save_dir = 'lstm_poetry_generation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

corpus_filename = 'robert_frost.txt'


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


class RobertFrostCorpus:

    def __init__(self, pretrained_word_embedding):

        self.pretrained = pretrained_word_embedding

        self.input_lines = []
        self.target_lines = []
        self.get_corpus()

        self.tokenizer = None
        self.vocab_size = 50000
        self.max_seq_len = 0
        self.input_sequences = None
        self.target_sequences = None
        self.tokenize()

        self.embedding = None
        self.build_word_embedding()

    def get_corpus(self):
        print(f'* calling {RobertFrostCorpus.get_corpus.__name__}')

        for line in open(os.path.join(save_dir, corpus_filename), encoding='utf8'):
            # remove trailing whitespace
            line = line.rstrip()
            # empty line
            if not line:
                continue

            # manipulate the original line to produce input and target line
            # we will use text(t) to predict text(t+1)
            # shift input by 1 step to produce target
            #
            # example:
            #
            #   t1    t2   t3  t4  t5    t6
            # <sos>  this  is  a  book  <eos>
            #
            #            t1   |   t2  |   t3  |   t4  |   t5
            # input  =  <sos> |  this |   is  |   a   |  book
            # target =  this  |   is  |   a   |  book |  <eos>
            #            t2       t3      t4      t5      t6
            input_line = '<sos> ' + line
            target_line = line + ' <eos>'

            self.input_lines.append(input_line)
            self.target_lines.append(target_line)

        print(f'=> total {len(self.input_lines)} input lines')
        print(f'=> total {len(self.target_lines)} target lines')

    def tokenize(self):
        print(f'* calling {RobertFrostCorpus.tokenize.__name__}')

        all_lines = self.input_lines + self.target_lines

        # set filters to '', else special characters would be removed,
        # and cause <sos> <eos> become sos eos
        self.tokenizer = Tokenizer(num_words=self.vocab_size,
                                   filters='')
        self.tokenizer.fit_on_texts(all_lines)
        self.input_sequences = self.tokenizer.texts_to_sequences(self.input_lines)
        self.target_sequences = self.tokenizer.texts_to_sequences(self.target_lines)

        self.max_seq_len = max(len(s) for s in self.input_sequences)
        print(f'=> max sequence length = {self.max_seq_len}')

        self.input_sequences = pad_sequences(self.input_sequences,
                                             maxlen=self.max_seq_len,
                                             padding='post')
        self.target_sequences = pad_sequences(self.target_sequences,
                                              maxlen=self.max_seq_len,
                                              padding='post')
        print(f'=> input sequences shape = {self.input_sequences.shape}')
        print(f'=> target sequences shape = {self.target_sequences.shape}')

        # one-hot targets

        # keras sparse categorical cross entropy loss won't work when each target is a sequence.
        # it only works when you only have one target per input.
        #
        # now each sample gives us an entire sequence of targets,
        # but keras sparse categorical cross entropy just wasn't written for that case.
        self.target_sequences = to_categorical(self.target_sequences, num_classes=self.vocab_size + 1)
        print(f'=> after one-hot encoded, target sequences shape = {self.target_sequences.shape}')

    def build_word_embedding(self):
        print(f'* calling {RobertFrostCorpus.build_word_embedding.__name__}')

        # using pretrained word embedding to build
        # word embedding for words in robert frost corpus

        # because keras tokenizer reserve index 0 for <pad>
        num_words = self.vocab_size + 1

        self.embedding = np.zeros((num_words, self.pretrained.embedding_size))

        # find a word vector for every word in tokenizer.word_index if i < vocab_size
        # word vector for <pad> is all zeros
        for word, i in self.tokenizer.word_index.items():
            if i < self.vocab_size:
                index = self.pretrained.word2idx.get(word)
                # word vector for a word not found in pretrained will be all zeros
                if index is not None:
                    self.embedding[i] = self.pretrained.embedding[index]
        print(f'=> get word embedding with shape = {self.embedding.shape}')


if __name__ == '__main__':

    glove = Glove(glove_filepath)
    corpus = RobertFrostCorpus(glove)

    # inputs

    inputs = Input(shape=(corpus.max_seq_len,))

    # layers

    embedding_layer = Embedding(input_dim=corpus.vocab_size + 1,
                                # no need to feed input_length here,
                                # because we'll adjust it from max_seq_len to 1 later
                                output_dim=glove.embedding_size,
                                weights=[corpus.embedding],
                                trainable=False)

    lstm_layer = LSTM(units=25,
                      return_sequences=True,
                      return_state=False)

    dense_layer = Dense(units=corpus.vocab_size + 1,
                        activation='softmax')

    # model

    x = embedding_layer(inputs)

    # input shape = (, steps=12, embedding_size=100)
    x = lstm_layer(x)
    # output shape = (, steps=12, units=25)

    # input shape = (, 12, 25)
    outputs = dense_layer(x)
    # output shape = (, 12, units=50001)

    #
    # this model is only for training!
    #
    # it inputs sequences [0 ~ T]
    # and outputs (predicts) sequences [1 ~ T + 1]
    #
    # the input length is fixed to T
    # but when we want to use this model to generate new text
    # we don't want to input sequence with length T
    # we just want to input one single word!
    #
    # => have to build a new model with input length 1 later
    #

    model = Model(inputs=inputs,
                  outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x=corpus.input_sequences,
              y=corpus.target_sequences,
              batch_size=128,
              epochs=100,
              validation_split=0.3)

    # make a new model for sampling

    # we'll only input one word at a time
    inputs2 = Input(shape=(1,))

    x = embedding_layer(inputs2)

    x = lstm_layer(x)

    outputs2 = dense_layer(x)

    sampling_model = Model(inputs=inputs2,
                           outputs=outputs2)

    word2idx = corpus.tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}

    def sample_line():

        # the first word to be fed into sampling model
        # its shape should be (batch_size, steps, )
        initial_input = np.array([[word2idx['<sos>']]])

        # if sampling model return this symbol, stop sampling
        eos_index = word2idx['<eos>']

        output_sentence = []
        for _ in range(corpus.max_seq_len):

            o = sampling_model.predict(initial_input)

            # o shape = (batch_size, steps, vocab_size)
            #         = (1, 1, vocab_size)
            probs = o[0, 0]

            if np.argmax(probs) == 0:
                print('<pad>')

            probs[0] = 0
            probs /= probs.sum()

            idx = np.random.choice(len(probs), p=probs)
            if idx == eos_index:
                break

            output_sentence.append(idx2word.get(idx, '<pad>'))

            initial_input[0, 0] = idx

        return ' '.join(output_sentence)

    while True:
        for i in range(4):
            print(sample_line())

        ans = input("---generate another? [Y/n]---")
        if ans and ans[0].lower().startswith('n'):
            break





