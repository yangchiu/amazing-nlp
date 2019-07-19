# for introduction to end-to-end memory networks, please see:
# https://www.paperweekly.site/papers/notes/181
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os

save_dir = 'memory_networks_yes_no_bot/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_filename = 'train_qa.bin'
test_filename = 'test_qa.bin'


class YesNoData():

    def __init__(self, save_dir, train_filename, test_filename):
        print(f'* calling {YesNoData.__init__.__name__}')

        self.train_data = None
        self.test_data = None
        self.get_data(save_dir, train_filename, test_filename)

        self.vocab_set = set()
        self.vocab_size = 0
        self.max_story_len = 0
        self.max_question_len = 0
        self.build_vocab_list()

        self.tokenizer = None
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

    yes_no_data = YesNoData(save_dir, train_filename, test_filename)
