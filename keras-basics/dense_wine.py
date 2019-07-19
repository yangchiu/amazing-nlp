from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.contrib.keras import models, layers


def get_data():
    print(f'* calling {get_data.__name__}')

    wine_data = load_wine()
    print(wine_data.DESCR)
    x_train, x_test, y_train, y_test = train_test_split(wine_data['data'],
                                                        wine_data['target'],
                                                        test_size=0.3)
    scaler = MinMaxScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)

    print(f'=> x_train shape = {scaled_x_train.shape}')
    print(f'=> x_test shape = {scaled_x_test.shape}')
    print(f'=> y_train shape = {y_train.shape}')
    print(f'=> y_test shape = {y_test.shape}')

    # => x_train shape = (124, 13)
    # => x_test shape = (54, 13)
    # => y_train shape = (124,)
    # => y_test shape = (54,)

    return scaled_x_train, scaled_x_test, y_train, y_test


if __name__ == '__main__':

    scaled_x_train, scaled_x_test, y_train, y_test = get_data()

    # model
    dnn_model = models.Sequential()

    # layers
    # input shape = (batch_size, input_dim=13) batch_size is handled by model.fit
    dnn_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
    # output shape = (batch_size, units=13)

    # input shape = (batch_size, input_dim=13) batch_size is handled by model.fit
    dnn_model.add(layers.Dense(units=13, activation='relu'))
    # output shape = (batch_size, units=13)

    # input shape = (batch_size, input_dim=13) batch_size is handled by model.fit
    dnn_model.add(layers.Dense(units=13, activation='relu'))
    # output shape = (batch_size, units=13)

    # input shape = (batch_size, input_dim=13) batch_size is handled by model.fit
    dnn_model.add(layers.Dense(units=3, activation='softmax'))
    # output shape = (batch_size, units=3)

    # loss, optimizer and metric
    dnn_model.compile(
        optimizer='adam',
        # if your targets are one-hot encoded, use categorical_crossentropy.
        # if your targets are integers, use sparse_categorical_crossentropy.
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train
    # batch_size can be assigned here. if unspecified, batch_size will default to 32.
    dnn_model.fit(scaled_x_train, y_train, epochs=50)

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
    predictions = dnn_model.predict_classes(scaled_x_test)
    print(classification_report(predictions, y_test))
