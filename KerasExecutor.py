import os
import pprint
import time
from math import sqrt, ceil, trunc

import pandas as pd
from numpy.ma import true_divide
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D
from keras.layers import Dropout, Convolution2D
from keras import callbacks
#from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf

print("Dispositivos disponibles:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



class KerasExecutor:
    # The number of neurons in the first and last layer included in network-structure is ommited.
    def __init__(self, dataset, test_size, metrics, early_stopping_patience, loss, first_data_column=1):

        self.dataset = dataset
        self.first_data_column = first_data_column
        self.test_size = test_size
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.loss = loss


        data = dataset["data"]

        self.x = true_divide(data, 255)

        y = dataset["target"]

        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_numeric = le.transform(y)
        self.y_hot_encoding = to_categorical(y_numeric).astype(float)


        self.n_in = len(self.x[1])
        self.n_out = len(self.y_hot_encoding[1])

    def execute(self, individual):

        #tf.reset_default_graph()
        K.clear_session()


        #print(str(individual.toString))
        #train, test, target_train, target_test = train_test_split(self.x, self.y_hot_encoding, test_size=self.test_size,
        #                                                          random_state=int(time.time()))
        train, test, target_train, target_test = train_test_split(self.x, self.y_hot_encoding, test_size=self.test_size,
                                                                  random_state=42)

        model = Sequential()

        list_layers_names = [l.type for l in individual.net_struct]
        print(",".join(list_layers_names))

        for index, layer in enumerate(individual.net_struct):

            if layer.type == "Dense":
                model.add(Dense(**layer.parameters))

            elif layer.type == "Dropout":
                model.add(Dropout(**layer.parameters))

            elif layer.type == "Convolution2D":
                layer.parameters['kernel_size'] = min(layer.parameters['kernel_size'], model.output_shape[1])
                #layer.parameters['nb_col'] = min(layer.parameters['nb_col'], model.output_shape[2])
                model.add(Convolution2D(**layer.parameters))

            elif layer.type == "MaxPooling2D":
                # Pool size checking...
                pool_size = (min(layer.parameters['pool_size'][0], model.output_shape[1]),
                             min(layer.parameters['pool_size'][1], model.output_shape[2]))

                if layer.parameters['strides'] is None:
                    model.add(MaxPooling2D(pool_size=pool_size, strides=(1,1)))
                else:
                    model.add(MaxPooling2D(pool_size=pool_size, strides=layer.parameters['strides']))

            elif layer.type == "Reshape":

                aspect_ratio = layer.parameters["target_shape"]

                if index == 0:
                    last_num_rows, last_num_cols = None, self.n_in
                else:
                    last_num_rows, last_num_cols = model.output_shape

                dividers = [k for k in range(2, int(sqrt(last_num_cols))) if last_num_cols % k == 0]
                num_columns = max(dividers)
                num_rows = int(last_num_cols / num_columns)

                if 'input_shape' in layer.parameters:

                    model.add(Reshape(target_shape=(num_rows, num_columns, 1),
                                      input_shape=layer.parameters["input_shape"]))
                else:
                    model.add(Reshape(target_shape=(num_rows, num_columns, 1)))

            elif layer.type == "Flatten":
                model.add(Flatten(**layer.parameters))

        # Train validation split
        train, validation, target_train, target_validation = train_test_split(train, target_train, test_size=0.2,
                                                                              random_state=42)

        model.compile(loss=self.loss, optimizer=individual.global_attributes.optimizer, metrics=self.metrics)

        # Stop criteria definition
        callbacks_array = [
            callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=self.early_stopping_patience, verbose=0,
                                    mode='max')]

        # Running model
        hist = model.fit(train, target_train, epochs=individual.global_attributes.epochs,
                         batch_size=individual.global_attributes.batch_size,
                         verbose=0, callbacks=callbacks_array, validation_data=(validation, target_validation)).__dict__


        scores_training = model.evaluate(train, target_train, verbose=0)
        scores_validation = model.evaluate(validation, target_validation, verbose=0)
        scores_test = model.evaluate(test, target_test, verbose=0)
        
        return model.metrics_names, scores_training, scores_validation, scores_test, model
