from models.cnn import find_cnn_model_params, fit_cnn_model, load_data_CNN
from models.rnn import find_rnn_model_params, fit_rnn_model, load_data_RNN
import time


x_train, y_train, x_test, y_test = load_data_CNN()

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
find_cnn_model_params(x_test, y_test)

#start_time = time.time()
#fit_cnn_model(x_train, y_train, x_test, y_test)
#print("--- %s seconds ---" % (time.time() - start_time))




x_train, y_train, x_test, y_test = load_data_RNN()

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
find_rnn_model_params(x_test, y_test)


#start_time = time.time()
#fit_rnn_model(x_train, y_train, x_test, y_test)
#print("--- %s seconds ---" % (time.time() - start_time))



    


