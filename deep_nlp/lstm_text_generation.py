import spacy
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding
from pickle import dump, load

save_dir = 'lstm_text_generation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

filename = 'moby_dick_four_chapters.txt' # 'melville-moby_dick.txt'

model_name = 'model.h5'
tokenizer_name = 'tokenizer'

class NovelData():

    def __init__(self, save_dir, filename):
        print(f'* calling {NovelData.__init__.__name__}')

        self.tokenized_corpus = None
        self.get_corpus(save_dir, filename)

        # 25 training words + 1 target word
        self.seq_len = 25 + 1
        self.sequences = []
        self.generate_sequences()

        self.tokenizer = None
        self.indexed_sequences = []
        self.vocab_size = 0
        self.word2idx()

        self.train_steps = 25
        self.target_len = 1
        self.x = None
        self.y = None
        self.x_y_split()

    def get_corpus(self, save_dir, filename):
        print(f'* calling {NovelData.get_corpus.__name__}')

        with open(os.path.join(save_dir, filename)) as f:
            corpus = f.read()

        # have to run "python -m spacy download en" first
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
        # if you don't change max_length manually,
        # you may encounter this error:
        # ValueError: [E088] Text of length 1198622 exceeds maximum of 1000000. The v2.x parser and NER models
        # require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may
        # cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase
        # the `deep_nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs
        # are too long by checking `len(text)`.
        nlp.max_length = 1198623
        # corpus would be tokenized after deep_nlp() operation
        self.tokenized_corpus = nlp(corpus)
        # remove punctuations
        self.tokenized_corpus = [
            token.text.lower() for token in self.tokenized_corpus
            if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
        ]
        print(f'=> length of tokenized corpus = {len(self.tokenized_corpus)}')

    def generate_sequences(self):
        print(f'* calling {NovelData.generate_sequences.__name__}')

        for i in range(self.seq_len, len(self.tokenized_corpus)):
            seq = self.tokenized_corpus[i-self.seq_len:i]
            self.sequences.append(seq)

        print(f'=> generate {len(self.sequences)} sequences')

    def word2idx(self):
        print(f'* calling {NovelData.word2idx.__name__}')

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.sequences)
        # !!! the word index starts from "1", instead of "0" !!!
        # print self.tokenizer.index_word to evaluate this
        #
        # so if you want to convert the indexed values of a word into one-hot encoded word class,
        # the number of classes should be len + 1, with empty class 0 (0 is preserved for <pad>)
        #
        # the input of texts_to_sequences should be a list of list with shape [num of words list][words list]
        # the output of texts_to_sequences is a list of list with shape [num of indexed words list][indexed words list]
        self.indexed_sequences = self.tokenizer.texts_to_sequences(self.sequences)

        self.vocab_size = len(self.tokenizer.word_counts)
        print(f'=> vocab size = {self.vocab_size}')

        # convert train sequences to numpy array
        self.indexed_sequences = np.array(self.indexed_sequences)
        print(f'=> sequences shape = {self.indexed_sequences.shape}')

    def x_y_split(self):
        print(f'* calling {NovelData.x_y_split.__name__}')

        # no need to one-hot encoding x, keras embedding layer take the raw indexed sequences
        self.x = self.indexed_sequences[:, :-self.target_len]
        self.y = self.indexed_sequences[:, -self.target_len]
        # one-hot encoding
        self.y = to_categorical(self.y, num_classes=self.vocab_size+1)

        print(f'=> x shape = {self.x.shape}')
        print(f'=> y shape = {self.y.shape}')

        # => x shape = (11312, 25)
        # => y shape = (11312, 2718)


def generate_text(model, tokenizer, max_input_seq_len, input_seq, num_gen_words):
    print(f'* calling {generate_text.__name__}')

    # we want to use our trained model,
    # feed (input_seq) into the model,
    # the max length of (input_seq) should less than (max_input_seq_len),
    # if len(input_seq) < (max_input_seq_len), it would be padded with "0"
    # if len(input_seq) > (max_input_seq_len), the head of (input_seq) would be truncated.
    #
    # once we feed (input_seq) into the model,
    # it would output one word.
    # the output word would be appended to (input_seq) to generate a new (input_seq),
    # and then feed the new (input_seq) into the model to output one more word.
    #
    # this process would be repeated for (num_gen_words) times to generate (num_gen_words) words.

    print(f'=> input sequence = {" ".join(input_seq)}')

    output_seq = []

    for i in range(num_gen_words):

        # texts_to_sequences input shape = [# of seq][input_seq]
        # texts_to_sequences output shape = [# of seq][indexed_input_seq]
        indexed_input_seq = tokenizer.texts_to_sequences([input_seq])

        # pad_sequences input shape = [# of seq][indexed_input_seq]
        # pad_sequences output shape = [# of seq][padded_indexed_input_seq]
        padded_indexed_input_seq = pad_sequences(indexed_input_seq,
                                                 maxlen=max_input_seq_len,
                                                 truncating='pre')

        # predict_classes input shape = [# of seq][padded_indexed_input_seq]
        # predict_classes output shape = [# of seq] (if return_sequences = False)
        predicted_word_index = model.predict_classes(padded_indexed_input_seq, verbose=0)[0]

        # convert from index back to word
        predicted_word = tokenizer.index_word[predicted_word_index]

        # append the predicted word to input_seq to generate new input_seq
        input_seq += [predicted_word]

        # store output_seq
        output_seq.append(predicted_word)

    print(f'=> output sequence = {" ".join(output_seq)}')


if __name__ == '__main__':

    novel_data = NovelData(save_dir, filename)

    if not os.path.exists(os.path.join(save_dir, model_name)) and \
       not os.path.exists(os.path.join(save_dir, tokenizer_name)):

        # because the index starts from "1", instead of "0"
        # a word is mapped to 1 ~ 17527
        # nothing is mapped to 0
        #
        # the actual word embedding input is from 0 ~ 17527
        vocab_size = novel_data.vocab_size + 1
        steps = novel_data.train_steps

        # model
        model = Sequential()

        # input shape = (batch_size, sequence_length=input_length=train_steps) batch_size is handled by model.fit
        model.add(
            Embedding(input_dim=vocab_size,
                      input_length=novel_data.train_steps,
                      output_dim=32)
        )
        # output shape = (batch_size, sequence_length, output_dim=32)

        # input shape = (batch_size, steps, input_dim=32) batch_size is handled by model.fit
        # units: dimensionality of the output space.
        # return_sequences: whether to return the last output in the output sequence, or the full sequence.
        model.add(LSTM(units=150,
                       return_sequences=True))
        # output shape = (batch_size, steps, units=150)

        # input shape = (batch_size, steps, input_dim=150) batch_size is handled by model.fit
        # only keep the last output
        model.add(LSTM(units=150))
        # output shape = (batch_size, units=150)

        # input shape = (batch_size, input_dim=150) batch_size is handled by model.fit
        model.add(Dense(units=150, activation='relu'))
        # output shape = (batch_size, units=150)

        # input shape = (batch_size, input_dim=150) batch_size is handled by model.fit
        model.add(Dense(units=vocab_size, activation='softmax'))
        # output shape = (batch_size, units=vocab_size)

        # loss, optimizer, metric
        model.compile(
            # if your targets are one-hot encoded, use categorical_crossentropy.
            # if your targets are integers, use sparse_categorical_crossentropy.
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        model.summary()

        # batch_size can be assigned here. if unspecified, batch_size will default to 32.
        model.fit(novel_data.x, novel_data.y, epochs=300, verbose=1)

        model.save(os.path.join(save_dir, model_name))
        dump(novel_data.tokenizer, open(os.path.join(save_dir, tokenizer_name), 'wb'))

    # load model
    model = load_model(os.path.join(save_dir, model_name))
    tokenizer = load(open(os.path.join(save_dir, tokenizer_name), 'rb'))

    # take a random input sequence seed
    length = len(novel_data.sequences)
    rand = np.random.randint(0, length)
    input_seq = novel_data.sequences[rand]

    # use the random input sequence to generate new sequence
    generate_text(model, tokenizer, novel_data.train_steps, input_seq, 50)
