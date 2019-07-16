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

    return scaled_x_train, scaled_x_test, y_train, y_test


if __name__ == '__main__':

    scaled_x_train, scaled_x_test, y_train, y_test = get_data()

    # model
    model = Sequential()
    model.add(Dense(units=8, input_dim=4, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    # loss, optimizer and metric
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # train
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
