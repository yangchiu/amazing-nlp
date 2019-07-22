# for introduction to end-to-end memory networks, please see:
# https://www.paperweekly.site/papers/notes/181

# the original paper:
# https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf

# summary of memory networks:
# u = embedding(question) shape = (max_question_len, n)
# m = embedding(story)    shape = (max_story_len, n)
# c = embedding(story)    shape = (max_story_len, max_question_len)
#
# p = softmax(u dot m)    shape = (max_story_len, max_question_len)
#
# o = p + c               shape = (max_story_len, max_question_len)
#
# a_hat = concat(oT, u)   shape = (max_question_len, max_story_len + n)
# a_hat = LSTM(a_hat)     shape = (max_story_len + n)
# a_hat = dense(a_hat)    shape = (vocab_size)
#
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, LSTM, Permute, Dropout
from keras.layers import add, dot, concatenate
import os
from pickle import dump, load

save_dir = 'simple_memory_network_yes_no_bot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_filename = 'train_qa.bin'
test_filename = 'test_qa.bin'

model_filename = 'yes_no_bot_model.h5'
tokenizer_name = 'tokenizer'


class YesNoData():

    def __init__(self, save_dir, train_filename, test_filename, tokenizer=None):
        print(f'* calling {YesNoData.__init__.__name__}')

        self.train_data = None
        self.test_data = None
        self.get_data(save_dir, train_filename, test_filename)

        self.vocab_set = set()
        self.vocab_size = 0
        self.max_story_len = 0
        self.max_question_len = 0
        self.build_vocab_list()

        self.tokenizer = tokenizer
        self.stories_test = None
        self.questions_test = None
        self.answers_test = None
        self.word2idx()

        self.stories_train, self.questions_train, self.answers_train = self.vectorize_data(self.train_data)
        self.stories_test, self.questions_test, self.answers_test = self.vectorize_data(self.test_data)

        print(f'=> sample story sequence: {self.stories_train[0]}')
        print(f'=> sample question sequence: {self.questions_train[0]}')
        print(f'=> sample one-hot encoded answer: {self.answers_train[0]}')

    def get_data(self, save_dir, train_filename, test_filename):
        print(f'* calling {YesNoData.get_data.__name__}')

        with open(os.path.join(save_dir, train_filename), 'rb') as f:
            self.train_data = pickle.load(f)
        with open(os.path.join(save_dir, test_filename), 'rb') as f:
            self.test_data = pickle.load(f)

    def build_vocab_list(self):
        print(f'* calling {YesNoData.build_vocab_list.__name__}')

        all_data = self.train_data + self.test_data

        for story, question, answer in all_data:
            self.vocab_set = self.vocab_set.union(set(story))
            self.vocab_set = self.vocab_set.union(set(question))
            # answer is just a string
            # add [] to transform it into a list and then a set
            self.vocab_set = self.vocab_set.union(set([answer]))

        # add an extra 1 to reserve "index 0" for keras's pad_sequences
        self.vocab_size = len(self.vocab_set) + 1

        print(f'=> vocab list = {self.vocab_set}')
        print(f'=> vocab size = {self.vocab_size}')

        self.max_story_len = max([len(data[0]) for data in all_data])
        self.max_question_len = max([len(data[1]) for data in all_data])

        print(f'=> max story length = {self.max_story_len}')
        print(f'=> max question length = {self.max_question_len}')

    def word2idx(self):
        print(f'* calling {YesNoData.word2idx.__name__}')

        # if filters are not assigned as [], punctuations would be removed in the tokenizer
        if not self.tokenizer:
            self.tokenizer = Tokenizer(filters=[])
            self.tokenizer.fit_on_texts(self.vocab_set)
        # Tokenizer will do lower() for you
        # and the word index starts from "1"
        print(f'=> word2idx = {self.tokenizer.word_index}')

    def vectorize_data(self, data):
        print(f'* calling {YesNoData.vectorize_data.__name__}')

        word2idx = self.tokenizer.word_index
        indexed_stories = []
        indexed_questions = []
        indexed_answers = []

        for story, question, answer in data:
            indexed_story = [word2idx[word.lower()] for word in story]
            indexed_question = [word2idx[word.lower()] for word in question]
            # add an extra 1 to reserve "index 0" for keras's pad_sequences
            one_hot_encoded_answer = np.zeros(len(word2idx) + 1)
            one_hot_encoded_answer[word2idx[answer]] = 1

            indexed_stories.append(indexed_story)
            indexed_questions.append(indexed_question)
            indexed_answers.append(one_hot_encoded_answer)

        # pre-padding is the default.
        # pass padding='pre' or 'post' to pad either before or after each sequence.
        indexed_stories = pad_sequences(indexed_stories, maxlen=self.max_story_len)
        indexed_questions = pad_sequences(indexed_questions, maxlen=self.max_question_len)
        indexed_answers = np.array(indexed_answers)

        return indexed_stories, indexed_questions, indexed_answers


if __name__ == '__main__':

    tokenizer = None
    if os.path.exists(os.path.join(save_dir, tokenizer_name)):
        tokenizer = load(open(os.path.join(save_dir, tokenizer_name), 'rb'))
    yes_no_data = YesNoData(save_dir, train_filename, test_filename, tokenizer)

    if not os.path.exists(os.path.join(save_dir, model_filename)):

        max_story_len = yes_no_data.max_story_len
        max_question_len = yes_no_data.max_question_len
        vocab_size = yes_no_data.vocab_size

        # placeholders
        stories = Input((max_story_len,))
        questions = Input((max_question_len,))

        # embedding A
        stories_embedding_A = Sequential()

        # input shape = (batch_size, max_story_len)
        # although the input_dim = vocab_size,
        # the input data is actually numerical, instead of one-hot encoded
        stories_embedding_A.add(Embedding(input_dim=vocab_size,
                                          input_length=max_story_len,
                                          output_dim=64))
        # output shape = (batch_size, max_story_len, output_dim=64)
        stories_embedding_A.add(Dropout(0.3))

        # embedding C
        stories_embedding_C = Sequential()

        # input shape = (batch_size, max_story_len)
        #
        # stories_embedding_C output_dim
        # should be equal to max_question_len
        # so it can be added to p later
        stories_embedding_C.add(Embedding(input_dim=vocab_size,
                                          input_length=max_story_len,
                                          output_dim=max_question_len))
        # output shape = (batch_size, max_story_len, output_dim=max_question_len)
        stories_embedding_C.add(Dropout(0.3))

        # embedding B
        questions_embedding_B = Sequential()

        # input shape = (batch_size, max_question_len)
        #
        # questions_embedding_B output_dim
        # should be equal to
        # stories_embedding_A output_dim
        #
        # so it can be dot-producted later
        questions_embedding_B.add(Embedding(input_dim=vocab_size,
                                            input_length=max_question_len,
                                            output_dim=64))
        # output shape = (batch_size, max_question_len, output_dim=64)
        questions_embedding_B.add(Dropout(0.3))

        # encoding using embeddings
        m = stories_embedding_A(stories)
        c = stories_embedding_C(stories)
        u = questions_embedding_B(questions)

        # inner product
        # (batch_size, max_story_len, output_dim=64) dot (batch_size, max_question_len, output_dim=64)
        # along axes [2, 2]
        # = (batch_size, max_story_len, max_question_len)
        #
        # the meaning is: is mi important to uj ?
        p_prime = dot([m, u], axes=[2, 2])
        p = Activation('softmax')(p_prime)

        # sum
        # (batch_size, max_story_len, max_question_len) add (batch_size, max_story_len, max_question_len)
        # = (batch_size, max_story_len, max_question_len)
        o = add([p, c])
        # reshape
        # from (batch_size, max_story_len, max_question_len)
        # to (batch_size, max_question_len, max_story_len)
        o = Permute((2, 1))(o)

        # concatenate
        # (batch_size, max_question_len, max_story_len) concat (batch_size, max_question_len, output_dim=64)
        # = (batch_size, max_question_len, max_story_len + output_dim)
        answer = concatenate([o, u])

        # input shape = (batch_size, max_question_len, max_story_len + output_dim)
        # return_sequences default value is False
        # so the steps dimension (max_question_len) is reduced
        answer = LSTM(units=32)(answer)
        # output shape = (batch_size, units)
        print(answer)

        answer = Dropout(0.5)(answer)

        # input shape = (batch_size, 32)
        answer = Dense(units=vocab_size)(answer)
        # output shape = (batch_size, vocab_size)

        # output a probability over the vocab
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model(inputs=[stories, questions], outputs=answer)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        stories_train = yes_no_data.stories_train
        questions_train = yes_no_data.questions_train
        answers_train = yes_no_data.answers_train
        stories_test = yes_no_data.stories_test
        questions_test = yes_no_data.questions_test
        answers_test = yes_no_data.answers_test

        model.fit(x=[stories_train, questions_train],
                  y=answers_train,
                  epochs=120,
                  validation_data=([stories_test, questions_test], answers_test)
                 )

        model.save(os.path.join(save_dir, model_filename))
        dump(yes_no_data.tokenizer, open(os.path.join(save_dir, tokenizer_name), 'wb'))

    # test the model
    model = load_model(os.path.join(save_dir, model_filename))
    tokenizer = load(open(os.path.join(save_dir, tokenizer_name), 'rb'))

    stories_test = yes_no_data.stories_test
    questions_test = yes_no_data.questions_test
    answers_test = yes_no_data.answers_test

    predicted_results = model.predict([stories_test, questions_test])

    print(f'*** test model ***')
    for i in range(10):
        print(f'------- set {i} --------')
        print(f'[story]')
        print(f'{tokenizer.sequences_to_texts([stories_test[i]])[0]}')
        print(f'[question]')
        print(f'{tokenizer.sequences_to_texts([questions_test[i]])[0]}')
        print(f'[real answer]')
        print(f'{tokenizer.index_word[np.argmax(answers_test[i])]}')
        print(f'[predicted answer]')
        print(f'{tokenizer.index_word[np.argmax(predicted_results[i])]}')
