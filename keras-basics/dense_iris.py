from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os

save_dir = 'dense_iris/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_data():
    print(f'* calling {get_data.__name__}')

    iris = load_iris()
    print(iris.DESCR)

    x = iris['data']

    # convert numerical classes into one-hot encoded
    y = to_categorical(iris['target'])

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3)

    scaler = MinMaxScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)

    print(f'=> x_train shape = {scaled_x_train.shape}')
    print(f'=> x_test shape = {scaled_x_test.shape}')
    print(f'=> y_train shape = {y_train.shape}')
    print(f'=> y_test shape = {y_test.shape}')

    # => x_train shape = (105, 4)
    # => x_test shape = (45, 4)
    # => y_train shape = (105, 3)
    # => y_test shape = (45, 3)

    return scaled_x_train, scaled_x_test, y_train, y_test


if __name__ == '__main__':

    scaled_x_train, scaled_x_test, y_train, y_test = get_data()

    # model
    model = Sequential()

    # input shape = (batch_size, input_dim=4) batch_size is handled by model.fit
    model.add(Dense(units=8, input_dim=4, activation='relu'))
    # output shape = (batch_size, units=8)

    # input shape = (batch_size, input_dim=8) batch_size is handled by model.fit
    model.add(Dense(units=8, activation='relu'))
    # output shape = (batch_size, units=8)

    # input shape = (batch_size, input_dim=8) batch_size is handled by model.fit
    model.add(Dense(units=3, activation='softmax'))
    # output shape = (batch_size, units=3)

    # loss, optimizer and metric
    model.compile(
        optimizer='adam',
        # if your targets are one-hot encoded, use categorical_crossentropy.
        # if your targets are integers, use sparse_categorical_crossentropy.
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # train
    # batch_size can be assigned here. if unspecified, batch_size will default to 32.
    model.fit(scaled_x_train, y_train, epochs=150, verbose=2)

    # test

    # model.predict will return the scores of the regression
    # model.predict_class will return the class of your prediction
    #
    # imagine you are trying to predict if the picture is a dog or a cat (you have a classifier):
    #
    # predict will return you: 0.6 cat and 0.2 dog (for example).
    # predict_class will return you cat
    #
    # Now imagine you are trying to predict house prices (you have a regressor):
    #
    # predict will return the predicted price
    # predict_class will not make sense here since you don't have a classifier
    #
    # TL:DR: use predict_class for classifiers (outputs are labels)
    # and use predict for regressions (outputs are non discrete)
    predictions = model.predict_classes(scaled_x_test)

    # predictions is numerical classes,
    # but y_test is one-hot encoded,
    # use argmax to convert one-hot encoded value back into numerical classes
    print(f'=> predictions results:')
    print(classification_report(y_test.argmax(axis=1), predictions))

    # save model
    model.save(os.path.join(save_dir, 'iris_model.h5'))

    new_model = load_model(os.path.join(save_dir, 'iris_model.h5'))

    new_predictions = new_model.predict_classes(scaled_x_test)

    print(f'=> load model and make predictions again:')
    print(classification_report(y_test.argmax(axis=1), new_predictions))
