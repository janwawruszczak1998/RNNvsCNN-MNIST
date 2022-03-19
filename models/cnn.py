import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical


from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix




def create_model(X, optimizer='SGD', loss='sparse_categorical_crossentropy'):

    model=Sequential()

    model.add(Conv2D(input_shape=X.shape[1:], filters=64, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
    
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model



def load_data_CNN():
    (x_train, y_train), (x_test, y_test) = load_data()

    # zmiana ksztaltu do tablicy  3-wymiarowej (wys=28, szer=28, liczba kanalow=1 (skala szarosci))
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)


    # konwersja do typu float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


    # normalizacja do RGB przez podzielenie przez RGBmax
    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test



def fit_cnn_model(x_train, y_train, x_test, y_test, optimizer='adam', loss='sparse_categorical_crossentropy', epochs=40, batch_size=64):
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    model = create_model(x_train, optimizer, loss)
    history = model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), batch_size=batch_size,
                        epochs=epochs)

    # wykresy acc i loss
    fig, ax = plt.subplots(2,1, figsize=(18, 10))
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('cnn_plot.png')

    # macierz konfuzji
    fig = plt.figure(figsize=(10, 10)) 

    y_pred = model.predict(x_test) # predykcja zakodowana jako np. 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    Y_pred = np.argmax(y_pred, 1) # zdekoduj do [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] => 2
    Y_test = np.argmax(y_test_cat, 1) 

    mat = confusion_matrix(Y_test, Y_pred) 

    sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.savefig('cnn_matrix.png')

    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_cnn.png', show_shapes=True, show_layer_names=True)

    return model


def find_cnn_model_params(X, labels):
    model = KerasClassifier(build_fn=create_model, X=X)

    param_grid = {
        'epochs': [10, 20, 40],
        'batch_size': [8, 16, 32, 64],
        'optimizer': ['rmsprop', 'adam', 'mse', 'SGD'],
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=StratifiedKFold(n_splits=5, random_state=1410, shuffle=True),
                        n_jobs=-1, return_train_score=True)
    grid_result = grid.fit(X, labels)

    df = pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False)
    
    
    with open("params_sorted_by_mean_cnn_model.txt", "a") as file:
        file.write(df.to_string())
        file.write("\n")
