import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
from sklearn.metrics import confusion_matrix
import os 
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
import keras_tuner as kt

from utils.model import *

leg_numerics_activities = {0: 'walking',
                  1: 'jogging',
                  2: 'stairs',
                  3: 'sitting',
                  4: 'standing',
                  5: 'kicking soccer ball',
                  6: 'playing catch tennis ball',
                  7: 'dribbling basket ball',
                 }

def load_iotensor():

    # Read the post-processing data
    train_features = pd.read_csv("assets/postprocessing_dataset/processed_pars_train_features.csv")
    train_labels = pd.read_csv("assets/postprocessing_dataset/processed_pars_train_labels.csv")
    valid_features = pd.read_csv("assets/postprocessing_dataset/processed_pars_validation_features.csv")
    valid_labels = pd.read_csv("assets/postprocessing_dataset/processed_pars_validation_labels.csv")
    test_features = pd.read_csv("assets/postprocessing_dataset/processed_pars_test_features.csv")
    test_labels = pd.read_csv("assets/postprocessing_dataset/processed_pars_test_labels.csv")

    # Change the data frame to tensor objects
    tensor_train_features = tf.convert_to_tensor(train_features)
    tensor_train_labels = tf.convert_to_tensor(train_labels)
    tensor_test_features = tf.convert_to_tensor(test_features)
    tensor_test_labels = tf.convert_to_tensor(test_labels)
    tensor_valid_features = tf.convert_to_tensor(valid_features)
    tensor_valid_labels = tf.convert_to_tensor(valid_labels)

    return tensor_train_features, tensor_train_labels, tensor_test_features, tensor_test_labels, tensor_valid_features, tensor_valid_labels

def tuning_dnn_model():

    tensor_train_features, tensor_train_labels, tensor_test_features, tensor_test_labels, tensor_valid_features, tensor_valid_labels = load_iotensor()
   
    tuner = kt.Hyperband(dnn_model_builder,
                     objective='val_accuracy',
                     max_epochs= 10,
                     factor=3,
                     directory='assets/hp_search',
                     project_name='dnn_result')

    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(tensor_train_features, tensor_train_labels, epochs=50, validation_data = (tensor_valid_features, tensor_valid_labels), callbacks=[stop_early])
    tuner.results_summary()
    # Return the tuner with best validation accuracy
    return tuner

def tuning_cnn_model():

    tensor_train_features, tensor_train_labels, tensor_test_features, tensor_test_labels, tensor_valid_features, tensor_valid_labels = load_iotensor()

    # reshape tensor to 2D
    tensor_train_features = tf.reshape(tensor_train_features, [-1, 64, 6])
    tensor_test_features = tf.reshape(tensor_test_features, [-1, 64, 6])
    tensor_valid_features = tf.reshape(tensor_valid_features, [-1, 64, 6])

    tuner = kt.Hyperband(cnn_model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='assets/hp_search',
                     project_name='cnn_result')

    stop_early = EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(tensor_train_features, tensor_train_labels, epochs=50, validation_data = (tensor_valid_features, tensor_valid_labels), callbacks=[stop_early])
    tuner.results_summary()
    # Return the tuner with best validation accuracy
    return tuner


def tuning_lstm_model():

    tensor_train_features, tensor_train_labels, tensor_test_features, tensor_test_labels, tensor_valid_features, tensor_valid_labels = load_iotensor()

    # reshape tensor to 2D
    tensor_train_features = tf.reshape(tensor_train_features, [-1, 64, 6])
    tensor_test_features = tf.reshape(tensor_test_features, [-1, 64, 6])
    tensor_valid_features = tf.reshape(tensor_valid_features, [-1, 64, 6])

    tuner = kt.Hyperband(lstm_model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='assets/hp_search',
                     project_name='lstm_result')

    stop_early = EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(tensor_train_features, tensor_train_labels, epochs=50, validation_data = (tensor_valid_features, tensor_valid_labels), callbacks=[stop_early])
    tuner.results_summary()
    # Return the tuner with best validation accuracy
    return tuner

def training_progress(history, model_name: str):
    """
        show the loss function and accuracy curve during training for training and validation dataset
        Args:
            history: contain values of loss and accuracy over epochs
        Returns:
    """
    warnings.filterwarnings(action = 'ignore')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    # Create subplots with a specific size
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, width 12 inches, height 6 inches

    # Plot the data on the first subplot (accuracy graph)
    ax[0].plot(epochs, acc, 'b', label='Training acc', color = 'red')
    ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
    ax[0].set_title('Training and validation acc')
    ax[0].legend()

    # Plot the loss graph
    ax[1].plot(epochs, loss, 'b', label='Training loss', color = 'red')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Check if the file exists and delete it if it does
    file_name = "assets/training_progress/" + model_name + "_progress" + ".png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)

def prediction_matrix(labels, predictions, model_name: str, data_type: str):
    """
        crete a confusion matrix which shows the predicted activities vs true activities
        Args:
            labels: tensor object of the true labels
            prediction: tensor object of the predicted labels
            model_name: name of the model (e.g. dnn, cnn, lstm)
            data_type: type of labels (train, validation, test)
    """
    # Change the probability matrix to labels prediction dataframe
    x = []
    num_leg_activities = 8
    for i in range(predictions.shape[0]):
        for j in range(num_leg_activities):
            if  max(predictions[i]) == predictions[i][j]: #find maximum prediction value in the softmax layer
                x.append(leg_numerics_activities[j])
    predicted_labels = pd.DataFrame({'predicted_labels':x})

    # change the labels from tensor to dataframe
    y = []
    for i in range(labels.shape[0]):
        y.append(leg_numerics_activities[labels[i].numpy()[0]])
    true_labels = pd.DataFrame({'predicted_labels':y})
    
    # Confusion matrix for classifying
    cm = confusion_matrix(y_true= true_labels, y_pred= predicted_labels)

    # Heatmap for confusion matrix
    plt.figure(figsize = (18,12))
    x_axis_labels = []
    y_axis_labels = []
    for i in range(8):
        x_axis_labels.append(leg_numerics_activities[i])
        y_axis_labels.append(leg_numerics_activities[i])
    g = sb.heatmap(cm, annot = True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt='.0f')
    g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 10)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Check if the file exists and delete it if it does
    file_name = "assets/prediction_results/" + f'{model_name}' + "/heatmap_" + f'{data_type}' + ".png"
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed outdated version of {file_name}")

    plt.savefig(file_name)


    
