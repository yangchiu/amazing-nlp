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
    return scaled_x_train, scaled_x_test, y_train, y_test


if __name__ == '__main__':

    scaled_x_train, scaled_x_test, y_train, y_test = get_data()

    # model
    dnn_model = models.Sequential()

    # layers
    dnn_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
    dnn_model.add(layers.Dense(units=13, activation='relu'))
    dnn_model.add(layers.Dense(units=13, activation='relu'))
    dnn_model.add(layers.Dense(units=3, activation='softmax'))

    # loss, optimizer and metric
    dnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train
    dnn_model.fit(scaled_x_train, y_train, epochs=50)

    # test
    predictions = dnn_model.predict_classes(scaled_x_test)
    print(classification_report(predictions, y_test))
