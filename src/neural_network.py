#!/usr/bin/python3

import numpy as np
from keras import layers, optimizers
from keras.models import Sequential, load_model

from src.neural_network_constants import NNConstants


def read_training_data():
    """
    Read training data into numpy arrays.

    :return: a 2D array containing features, each row is a sample, each column is a feature;
             a 1D array containing labels
    """
    # read and normalize training data
    x_train = np.array([[]])
    y_train = np.array([])

    print("Shape of x_train:", np.shape(x_train))
    print("Shape of y_train:", np.shape(y_train))

    return x_train, y_train


def read_test_data():
    """
    Read test data into numpy arrays.

    :return: a 2D array containing features, each row is a sample, each column is a feature;
             a 1D array containing labels
    """
    # read and normalize test data
    x_test = np.array([[]])
    y_test = np.array([])

    print("Shape of x_test:", np.shape(x_test))
    print("Shape of y_test:", np.shape(y_test))

    return x_test, y_test


def get_model():
    """
    Define and return a neural network model.
    NOTE: here we define a regression neural network with two hidden layers as an example.

    :return: a Keras neural network model
    """
    model = Sequential()

    # first hidden layer
    model.add(layers.Dense(NNConstants.hidden_layer_1_size, input_dim=NNConstants.num_features))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Dropout(NNConstants.dropout_rate))

    # second hidden layer
    model.add(layers.Dense(NNConstants.hidden_layer_2_size))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Dropout(NNConstants.dropout_rate))

    # output layer
    model.add(layers.Dense(1, activation="sigmoid"))

    # define optimizer and loss function; compile model
    optimizer = optimizers.Adam(learning_rate=NNConstants.learning_rate)
    model.compile(loss="mean_absolute_error", optimizer=optimizer)

    # print model summary
    model.summary()

    return model


def train(model_path):
    """
    Train neural network, and save the model.

    :param model_path: str, the model file path to save to
    :return: None
    """
    # read training data
    x_train, y_train = read_training_data()

    # define callbacks
    callbacks = []

    # get and train model
    model = get_model()
    model.fit(x_train, y_train,
              validation_split=NNConstants.validation_split,
              batch_size=NNConstants.batch_size,
              epochs=NNConstants.epochs,
              callbacks=callbacks)

    # save model
    model.save(model_path)


def test(model_path):
    """
    Predict labels for test data, and calculate loss.

    :param model_path: str, the model file path to load from
    :return: None
    """
    # read test data
    x_test, y_test = read_test_data()

    # load model
    model = load_model(model_path)

    # predict labels
    pred = model.predict(x_test).T[0]

    # calculate loss
    # NOTE: we use mean absolute error here as an example
    loss = np.mean(np.abs(y_test - pred))
    print("Loss:", loss)


def neural_network_wrapper(task, model_path):
    """
    Call subroutines to train/test neural network.

    :param task: int, the task to run
    :param model_path: str, the model file path
    :return: None
    """
    if task == 1:
        train(model_path=model_path)
    elif task == 2:
        test(model_path=model_path)
    elif task == 3:
        train(model_path=model_path)
        test(model_path=model_path)
    else:
        get_model()
