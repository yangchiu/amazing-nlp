# Amazing NLP
- walk through Tensorflow/Keras basics
- walk through all modern deep NLP models
- eventually we'll build an universal deep NLP toolkits for Chinese

#### [A] tensorflow-basics
1. [linear_regression.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/linear_regression.py)
> use simple linear regression to fit a line
2. [logistic_regression_mnist.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/logistic_regression_mnist.py)
> apply logistic regression to MNIST digit recognition
3. [cnn_mnist.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/cnn_mnist.py)
> apply convolution neural network to MNIST digit recognition
4. [cnn_cifar10.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/cnn_cifar10.py)
> apply convolution neural network to CIFAR-10 object recognition
5. [simple_rnn_reconstruct_sequences.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/simple_rnn_reconstruct_sequences.py)
> use recurrent neural network to reconstruct sine wave
6. [multi_lstm_time_series.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/multi_lstm_time_series.py)
> apply multi-layers LSTM to milk production time-series prediction
7. [autoencoder_pca_simple.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/autoencoder_pca_simple.py)
> use autoencoder as PCA to perform dimension reduction from 3D to 2D
8. [autoencoder_pca_30to2.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/autoencoder_pca_30to2.py)
> use autoencoder as PCA to perform dimension reduction from 30D to 2D
9. [stacked_autoencoder_reconstruct_mnist.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/stacked_autoencoder_reconstruct_mnist.py)
> use autoencoder to generate new training data for MNIST
10. [gan.py](https://github.com/yangchiu/amazing-nlp/blob/master/tensorflow-basics/gan.py)
> use genarative adversarial network to generate new training data for MNIST

#### [B] keras-basics
1. [dense_wine.py](https://github.com/yangchiu/amazing-nlp/blob/master/keras-basics/dense_wine.py)
> apply fully-connected network to wine classification with sparse categorical cross entropy
2. [dense_iris.py](https://github.com/yangchiu/amazing-nlp/blob/master/keras-basics/dense_iris.py)
> apply fully-connected network to iris classification with categorical cross entropy
3. [bilstm_investigation.py](https://github.com/yangchiu/amazing-nlp/blob/master/keras-basics/bilstm_investigation.py)
> investigate the relations between output, hidden states, cell states of LSTM and BiLSTM and their shapes
4. [gru_investigation.py](https://github.com/yangchiu/amazing-nlp/blob/master/keras-basics/gru_investigation.py)
> investigate the relations between output and hidden states of GRU and BiGRU and their shapes
     
#### [C] nlp-basics
1. [pretrained_word2vec.py](https://github.com/yangchiu/amazing-nlp/blob/master/nlp-basics/pretrained_word2vec.py)
> load pre-trained word2vec model into gensim
2. [pretrained_glove.py](https://github.com/yangchiu/amazing-nlp/blob/master/nlp-basics/pretrained_glove.py)
> load pre-trained Glove model into numpy array
3. [bow_classifier.py](https://github.com/yangchiu/amazing-nlp/blob/master/nlp-basics/bow_classifier.py)
> use bag-of-words and pre-trained Glove to perform text classification
     
#### [D] deep-nlp
1. [word2vec.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/word2vec.py)
> train a new word2vec model using Tensorflow's nce-loss
2. [word2vec_skip_gram_negative_sampling.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/word2vec_skip_gram_negative_sampling.py)
> train a new word2vec model using Tensorflow, but doesn't use nce-loss, do the negative sampling ourselves instead
3. [word2vec_gensim.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/word2vec_gensim.py)
> use gensim API to train a new word2vec model
4. [glove.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/glove.py)
> implement and train a new Glove model using Tensorflow
5. [ner_tf.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/ner_tf.py)
> perform name-entity recognition using Tensorflow
6. [ner_keras.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/ner_keras.py)
> perform name-entity recognition using Keras
7. [lstm_text_generation.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/lstm_text_generation.py)
> train a language model based on Moby Dick, and generate new text sequences using this model
8. [lstm_comments_classification.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/lstm_comments_classification.py)
> use LSTM and pre-trained Glove to perform text classification
9. [cnn_comments_classification.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/cnn_comments_classification.py)
> prove that not only RNN, CNN can be used on time-series data and perform text classification, too
10. [simple_memory_network_yes_no_bot.py](https://github.com/yangchiu/amazing-nlp/blob/master/deep-nlp/simple_memory_network_yes_no_bot.py)
> apply a simple memory network to bAbI dataset, build a model can give a yes/no answer based on the given story and question
11. [dual_lstm_mnist.py]()
> prove that not only CNN, RNN can be used on image data and perform digit recognition, too
     