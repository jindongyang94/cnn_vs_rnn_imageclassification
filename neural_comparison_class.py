"""
The idea of this class is to allow proper encapsulation of all neural network functionalities
"""
# Imports
import sys
import os

import cv2
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import SimpleRNN
from keras.layers import Dropout, Flatten, Activation
from keras.models import load_model
from keras.optimizers import RMSprop
from keras import initializers
from keras import backend as K
from sklearn.metrics import roc_auc_score


class NeuralNetworkClassifier(object):
    """
    Class to encapsulate a conventional neural network training and evaluation model.
    Each instantiation should only load a single model instance
    with a single data choice, instead of switching models to and fro.
    """

    def __init__(self, chosen_model='cnn', data_choice='fashion'):
        # Internal Processes Trigger
        self._data_loaded = False
        self._data_preprocessed = False
        self._trained = False

        # Internal Model Choices
        self._model_choices = ['cnn', 'rnn']
        if chosen_model in self._model_choices:
            self._chosen_model = chosen_model
        else:
            errmsg = "[!] Error: Model Specification needs to be within the choices of" + str(self._model_choices)
            print(errmsg, file=sys.stderr)

        self._data_choices = ['fashion', 'flowers']
        if data_choice in self._data_choices:
            self._data_chosen = data_choice
        else:
            errmsg = "[!] Error: Model Specification needs to be within the choices of" + str(self._model_choices)
            print(errmsg, file=sys.stderr)

    def __load_data(self):
        if self._data_chosen == 'fashion_mnist':
            self.__load_data_fashion()
        elif self._data_chosen == "flowers":
            self.__load_data_flowers()

    def __load_data_fashion(self):
        # the data, shuffled and split between train and test sets
        (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.fashion_mnist.load_data()
        self.num_classes = 10
        self._data_loaded = True

    def __load_data_flowers(self):
        data_dir = "./datasets/flowers/"
        folders = os.listdir(data_dir)

        image_names = []
        self.y_train = []
        self.x_train = []

        size = 64, 64

        for folder in folders:
            for file in os.listdir(os.path.join(data_dir, folder)):
                if file.endswith("jpg"):
                    image_names.append(os.path.join(data_dir, folder, file))
                    self.y_train.append(folder)
                    img = cv2.imread(os.path.join(data_dir, folder, file))
                    im = cv2.resize(img, size)
                    self.x_train.append(im)
                else:
                    continue

        self.num_classes = 5
        self._data_loaded = True

    def __preprocessing(self):
        if not self._data_loaded:
            self.__load_data()

        if self._data_chosen == 'fashion_mnist':
            self.__preprocessing_fashion()
        elif self._data_chosen == "flowers":
            self.__preprocessing_flowers()

    def __preprocessing_fashion(self):
        if self._chosen_model == 'cnn':
            # input image dimensions
            img_rows, img_cols = 28, 28

            if K.image_data_format() == 'channels_first':
                self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
                self.x_val = self.x_val.reshape(self.x_val.shape[0], 1, img_rows, img_cols)
            else:
                self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
                self.x_val = self.x_val.reshape(self.x_val.shape[0], img_rows, img_cols, 1)

        elif self._chosen_model == 'rnn':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1, 1)
            self.x_val = self.x_val.reshape(self.x_val.shape[0], -1, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_val = self.x_val.astype('float32')
        self.x_train /= 255
        self.x_val /= 255
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_val.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
        self._data_preprocessed = True

    def __preprocessing_flowers(self):
        self.x_train = np.array(self.x_train)
        # Reduce the RGB values between 0 and 1
        self.x_train = self.x_train.astype('float32') / 255.0

        # Converting labels into numbers
        self.y_train_categories = self.y_train
        label_dummies = pd.get_dummies(self.y_train)
        self.y_train = label_dummies.values.argmax(1)

        # Shuffle the labels and images randomly for better results
        union_list = list(zip(self.x_train, self.y_train))
        random.shuffle(union_list)
        self.x_train, self.y_train = zip(*union_list)

        # Convert the shuffled list to numpy array type
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train,
                                                                              self.y_train, test_size=0.2,
                                                                              random_state=42)
        # Reshape model if chosen model is RNN
        if self._chosen_model == 'rnn':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1, 1)
            self.x_val = self.x_val.reshape(self.x_val.shape[0], -1, 1)

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_val.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
        self._data_preprocessed = True

    def __create_model(self):
        if not self._data_preprocessed:
            self.__preprocessing()

        if self._chosen_model == 'cnn':
            self.__create_model_cnn()
        elif self._chosen_model == 'rnn':
            self.__create_model_rnn()

    def __create_model_cnn(self):
        self.batch_size = 128
        self.n_epochs = 12

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=self.x_train.shape[1:]))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        # keras.optimizers.Adadelta()
        print(self.model.summary())

    def __create_model_rnn(self):
        self.batch_size = 32  # Size of each batch
        # self.n_epochs = 200
        self.n_epochs = 12
        hidden_units = 100
        # learning_rate = 1e-6
        # clip_norm = 1.0
        self.model = Sequential()
        self.model.add(SimpleRNN(hidden_units,
                                 kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                 recurrent_initializer=initializers.Identity(gain=1.0),
                                 activation='relu',
                                 input_shape=self.x_train.shape[1:]))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))
        # rmsprop = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        print(self.model.summary())

    def train(self, save_model=False):
        if not self._data_loaded:
            self.__load_data()
        if not self._data_preprocessed:
            self.__preprocessing()
        self.__create_model()

        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size, epochs=self.n_epochs, verbose=1)
        # validation_data = (self.x_val, self.y_val)
        self._trained = True

        model_path = "./saved_models/" + self._chosen_model + "-model.h5"
        if save_model:
            self.model.save(model_path)
        return self.model

    def evaluate(self, model=None):
        if not self._trained and not model:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        model = load_model(model) if model else self.model
        score = model.evaluate(self.x_val, self.y_val, verbose=0)
        y_predict = model.predict(self.x_val)
        performance_metrics = {
            "Test Loss": score[0],
            "Test Accuracy": score[1],
            "Test AUC Score": roc_auc_score(self.y_val, y_predict, average='weighted')
        }
        return performance_metrics


if __name__ == "__main__":
    # cnn_classifier = NeuralNetworkClassifier(chosen_model='cnn', data_choice='flowers')
    # cnn_classifier.train(save_model=True)
    # cnn_metrics = cnn_classifier.evaluate()

    rnn_classifier = NeuralNetworkClassifier(chosen_model='rnn', data_choice='flowers')
    rnn_classifier.train(save_model=True)
    rnn_metrics = rnn_classifier.evaluate()

    # print('CNN Classifier Metrics')
    # print(cnn_metrics)
    print('RNN Classifier Metrics')
    print(rnn_metrics)
