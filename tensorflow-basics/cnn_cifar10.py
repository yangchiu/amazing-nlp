import pickle
import numpy as np

cifar_dir = 'cifar-10-batches-py'
cifar_filenames = ['batches.meta', 'data_batch_1',
                   'data_batch_2', 'data_batch_3',
                   'data_batch_4', 'data_batch_5',
                   'test_batch']

class CifarHelper():

    def __init__(self, cifar_dir, cifar_filenames):

        self.i = 0

        self.train_batch = []
        self.test_batch = []

        self.label_names = None

        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None

        self.get_data(cifar_dir, cifar_filenames)
        self.setup_images_and_labels()


    def get_data(self, cifar_dir, cifar_filenames):
        print(f'* calling {CifarHelper.get_data.__name__}')

        def unpickle(filename):
            with open(filename, 'rb') as f:
                cifar_data = pickle.load(f, encoding='bytes')
            return cifar_data

        batch_meta = unpickle(f'{cifar_dir}/{cifar_filenames[0]}')
        self.label_names = batch_meta[b'label_names']
        print(f'=> get cifar label names:')
        print(f'   {self.label_names}')

        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[1]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[2]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[3]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[4]}'))
        self.train_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[5]}'))
        print(f'=> get cifar train batch:')
        print(f'   {self.train_batch[0].keys()}')

        self.test_batch.append(unpickle(f'{cifar_dir}/{cifar_filenames[6]}'))
        print(f'=> get cifar test batch:')
        print(f'   {self.test_batch[0].keys()}')

    def setup_images_and_labels(self):
        print(f'* calling {CifarHelper.setup_images_and_labels.__name__}')

        print(f'=> setting up training images and labels')
        # train_batch[i][b'data'] is of type "numpy.ndarray" with shape (10000, 3072)
        # train_batch[i][b'labels'] is of type "list" with length 10000

        # concat 5 batches data of shape (10000, 3072) using vstack
        # the output is of shape (50000, 3072)
        self.training_images = np.vstack(batch[b'data'] for batch in self.train_batch)
        print(f'=> shape of training_images: {self.training_images.shape}')
        self.training_images = self.training_images.reshape(
            int(self.training_images.shape[0]),
            3,
            32,
            32
        ).transpose(0, 2, 3, 1) # the order should be (batch_size, height, width, channels)
        print(f'=> reshape training_images to: {self.training_images.shape}')

        # concat 5 batches labels of length 10000 using hstack
        # the output is of shape (50000,)
        self.training_labels = np.hstack(batch[b'labels'] for batch in self.train_batch)
        print(f'=> shape of training_labels: {self.training_labels.shape}')
        self.training_labels = self.one_hot_encode(self.training_labels, 10)

        print(f'=> setting up testing images and labels')

        self.test_images = np.vstack(batch[b'data'] for batch in self.test_batch)
        print(f'=> shape of test_images: {self.test_images.shape}')
        #self.test_images = self.test_images


        #self.training_images
        print(self.train_batch[0][b'data'].shape)
        print(len(self.train_batch[0][b'labels']))

    def one_hot_encode(self, vector, classes=10):
        print(f'* calling {CifarHelper.one_hot_encode.__name__}')

        length = len(vector)
        out = np.zeros((length, classes))
        out[range[length], vector] = 1
        return out


if __name__ == '__main__':
    CifarHelper(cifar_dir, cifar_filenames)