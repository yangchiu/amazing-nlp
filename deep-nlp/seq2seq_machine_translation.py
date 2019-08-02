#
# it's an encoder - decoder architecture
#
# input sequence would be fed into encoder,
# the last hidden state of encoder would be fed into decoder as hidden state
#
# decoder generate outputs based on encoder's hidden state and input
#
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

save_dir = 'seq2seq_machine_translation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# download the data at: http://www.manythings.org/anki/
data_filename = 'eng2spa.txt'


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


class Eng2SpaData:

    def __init__(self, pretrained_word_embedding):
        print(f'* calling {Eng2SpaData.__init__.__name__}')

        self.pretrained = pretrained_word_embedding

        # restrict total numbers of samples
        self.num_samples = 10000
        self.encoder_inputs = []
        self.decoder_targets = []
        self.decoder_inputs_teacher_forcing = []
        self.get_data()

        self.max_vocab_size = 20000
        self.encoder_vocab_size = 0
        self.decoder_vocab_size = 0
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.encoder_max_seq_len = 0
        self.decoder_max_seq_len = 0
        self.tokenize()

        self.embedding = None
        self.build_word_embedding()

    def get_data(self):
        print(f'* calling {Eng2SpaData.get_data.__name__}')

        n = 1
        for line in open(os.path.join(save_dir, data_filename), encoding='utf8'):

            # restrict total numbers of samples
            if n > self.num_samples:
                break

            # the data should be in [english_sentence \t spanish_sentence] format
            if '\t' not in line:
                continue

            encoder_input, target = line.rstrip().split('\t')

            # encoder input is just for the last hidden state
            # no need to add <sos> or <eos>
            encoder_input = encoder_input

            # t + 1 decoder target
            decoder_target = target + ' <eos>'

            # t decoder input, for teacher forcing
            decoder_input_teacher_forcing = '<sos> ' + target

            self.encoder_inputs.append(encoder_input)
            self.decoder_targets.append(decoder_target)
            self.decoder_inputs_teacher_forcing.append(decoder_input_teacher_forcing)

            n += 1

        print(f'=> get {len(self.encoder_inputs)} samples')
        self.num_samples = len(self.encoder_inputs)

        print(f'=> sample:')
        print(f'=> encoder input sample: {self.encoder_inputs[100]}')
        print(f'=> decoder target sample: {self.decoder_targets[100]}')
        print(f'=> decoder input teacher forcing: {self.decoder_inputs_teacher_forcing[100]}')

    def tokenize(self):
        print(f'* calling {Eng2SpaData.tokenize.__name__}')

        self.encoder_tokenizer = Tokenizer(num_words=self.max_vocab_size)
        self.encoder_tokenizer.fit_on_texts(self.encoder_inputs)
        self.encoder_inputs = self.encoder_tokenizer.texts_to_sequences(self.encoder_inputs)

        print(f'=> found {len(self.encoder_tokenizer.word_index)} unique encoder input words')
        self.encoder_vocab_size = min(len(self.encoder_tokenizer.word_index) + 1, self.max_vocab_size + 1)

        self.encoder_max_seq_len = max(len(s) for s in self.encoder_inputs)
        print(f'=> encoder max sequence length = {self.encoder_max_seq_len}')

        # set filters to '' to prevent filter out <sos> and <eos>
        self.decoder_tokenizer = Tokenizer(num_words=self.max_vocab_size, filters='')
        self.decoder_tokenizer.fit_on_texts(self.decoder_targets + self.decoder_inputs_teacher_forcing)
        self.decoder_targets = self.decoder_tokenizer.texts_to_sequences(self.decoder_targets)
        self.decoder_inputs_teacher_forcing = self.decoder_tokenizer.texts_to_sequences(self.decoder_inputs_teacher_forcing)

        print(f'=> found {len(self.decoder_tokenizer.word_index)} unique decoder output words')
        self.decoder_vocab_size = min(len(self.decoder_tokenizer.word_index) + 1, self.max_vocab_size + 1)

        self.decoder_max_seq_len = max(len(s) for s in self.decoder_targets)
        print(f'=> decoder max sequence length = {self.decoder_max_seq_len}')

        self.encoder_inputs = pad_sequences(self.encoder_inputs, maxlen=self.encoder_max_seq_len)
        print(f'=> encoder inputs shape = {self.encoder_inputs.shape}')
        print(f'=> encoder inputs sample = {self.encoder_inputs[0]}')

        self.decoder_targets = pad_sequences(self.decoder_targets, maxlen=self.decoder_max_seq_len, padding='post')
        print(f'=> decoder targets shape = {self.decoder_targets.shape}')
        print(f'=> decoder targets sample = {self.decoder_targets[0]}')

        self.decoder_inputs_teacher_forcing = pad_sequences(self.decoder_inputs_teacher_forcing,
                                                            maxlen=self.decoder_max_seq_len,
                                                            padding='post')
        print(f'=> decoder inputs teacher forcing shape = {self.decoder_inputs_teacher_forcing.shape}')
        print(f'=> decoder inputs teacher forcing sample = {self.decoder_inputs_teacher_forcing[0]}')

        # one-hot encoding decoder_targets
        # since we cannot use sparse categorical cross entropy when we have sequences
        self.decoder_targets = to_categorical(self.decoder_targets, self.decoder_vocab_size)
        print(f'=> after one-hot encoded, decoder targets shape = {self.decoder_targets.shape}')

        # no need to one-hot encoding decoder_inputs_teacher_forcing
        # it would be fed into embedding layer directly

    def build_word_embedding(self):
        print(f'* calling {Eng2SpaData.build_word_embedding.__name__}')

        # using pretrained word embedding to build
        # word embedding for words in eng2spa corpus

        # because keras tokenizer reserve index 0 for <pad>
        num_words = self.encoder_vocab_size

        self.embedding = np.zeros((num_words, self.pretrained.embedding_size))

        # find a word vector for every word in tokenizer.word_index if i < vocab_size
        # word vector for <pad> is all zeros
        for word, i in self.encoder_tokenizer.word_index.items():
            if i < self.encoder_vocab_size:
                index = self.pretrained.word2idx.get(word)
                # word vector for a word not found in pretrained will be all zeros
                if index is not None:
                    self.embedding[i] = self.pretrained.embedding[index]
        print(f'=> get word embedding with shape = {self.embedding.shape}')

        # only the encoder inputs (english) has pretrained word embedding
        # we don't build pretrained word embeddings for decoder inputs/targets (spanish)
        # but don't worry, just train a new word embeddings


if __name__ == '__main__':

    glove = Glove(glove_filepath)
    eng2spa = Eng2SpaData(glove)

    hidden_dim = 256
    batch_size = 64
    epochs = 100

    # encoder

    encoder_inputs_placeholder = Input(shape=(eng2spa.encoder_max_seq_len, ))

    encoder_x = Embedding(input_dim=eng2spa.encoder_vocab_size,
                          input_length=eng2spa.encoder_max_seq_len,
                          output_dim=glove.embedding_size,
                          weights=[eng2spa.embedding])(encoder_inputs_placeholder)

    encoder_o, encoder_h, encoder_c = LSTM(units=hidden_dim,
                                           # hidden state would be fed into decoder
                                           # so we have to set return_state = True
                                           return_state=True)(encoder_x)

    # keep states, and pass them into decoder
    encoder_states = [encoder_h, encoder_c]

    # decoder

    decoder_inputs_placeholder = Input(shape=(eng2spa.decoder_max_seq_len, ))

    _decoder_embedding_layer = Embedding(input_dim=eng2spa.decoder_vocab_size,
                                         # don't have to set input_length here
                                         # because input_length would be changed
                                         # from max_seq_len to 1
                                         # when we create the sampling model
                                         output_dim=hidden_dim
                                         )

    decoder_x = _decoder_embedding_layer(decoder_inputs_placeholder)

    # since decoder is many-to-many model,
    # we have to set return_sequences = True
    _decoder_lstm = LSTM(units=hidden_dim,
                         return_sequences=True)

    decoder_output = _decoder_lstm(decoder_inputs_placeholder,
                                   initial_state=)