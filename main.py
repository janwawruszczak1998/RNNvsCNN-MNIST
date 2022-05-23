import numpy

from models.cnn import find_cnn_model_params, fit_cnn_model, load_data_CNN
from models.rnn import find_rnn_model_params, fit_rnn_model, load_data_RNN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
import time
import numpy as np

def revert_categorical(X_as_categorical) -> np.ndarray:
    return np.argmax(X_as_categorical, axis=1)

# x_train, y_train, x_test, y_test = load_data_CNN()
#
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])
#find_cnn_model_params(x_test, y_test)

# start_time = time.time()
# fit_cnn_model(x_train, y_train, x_test, y_test)
# print("--- %s seconds ---" % (time.time() - start_time))


# x_train, y_train, x_test, y_test = load_data_RNN()
#
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])
#find_rnn_model_params(x_test, y_test)

# start_time = time.time()
# fit_rnn_model(x_train, y_train, x_test, y_test)
# print("--- %s seconds ---" % (time.time() - start_time))

(x_train, y_train), (x_test, y_test) = load_data()

X = np.concatenate((x_test, x_train), axis=0)
y = np.concatenate((y_test, y_train), axis=0)

print(X.shape, y.shape)

folds = 5
kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1410)

results = np.zeros((2, folds))

X_cnn = X.reshape(-1, 28, 28, 1).astype('float32')
X_cnn /= 255.0
X_rnn = X.astype('float32')

for id, (train, test) in enumerate(kfold.split(X, y)):
    print(id)
    cnn = fit_cnn_model(X_cnn[train], y[train], id)
    y_pred = cnn.predict(X_cnn[test])
    results[0, id] = accuracy_score(y[test], revert_categorical(y_pred))
    print(results[0, id])

    rnn = fit_rnn_model(X_rnn[train], y[train], id)
    y_pred = rnn.predict(X_rnn[test])
    results[1, id] = accuracy_score(y[test], revert_categorical(y_pred))

    print(results)
    np.save(file="results", arr=results)


