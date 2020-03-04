import numpy as np
import pandas as pd
import shap
# load JS visualization code to notebook
shap.initjs()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import model_from_json

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
os.chdir("../")

from config.autoencoder_specifications import *


def autoencoder(X_train, X_test):
    input_shape = X_train.shape[1]
    input_layer = Input(shape = (input_shape,))
    hid_layer1 = Dense(dim_hid1, activation = activation, name = 'hid_layer1')(input_layer)
    hid_layer2 = Dense(dim_hid2, activation = activation, name = 'hid_layer2')(hid_layer1)
    hid_layer3 = Dense(dim_hid3, activation = activation, name = 'hid_layer3')(hid_layer2)
    output_layer = Dense(input_shape)(hid_layer3)
    model = Model(input = input_layer, output = output_layer)
    optimiser = Adam(lr = learning_rate)
    model.compile(optimizer = optimiser, loss = 'mean_squared_error')
    model.fit(x = X_train, y = X_train, batch_size = batch_size, shuffle = True,
              epochs = epochs, verbose = 0, validation_data = [X_test, X_test],
             callbacks = [early_stop])
    return model
    

def main():
    X, y = shap.datasets.boston()
    print("Using Boston Housing Dataset as an example but only using housing data not pricing data.")
    
    ## Standardise the data and split the training data and test data
    std = StandardScaler()
    X_standard = std.fit_transform(X)
    X_train, X_test = train_test_split(X_standard)
    
    ## Train Autoencoder
    model = autoencoder(X_train, X_test)
    print(model.summary())
    
    
    ## Save the Autoencoder
    model_json = model.to_json()
    with open("log/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("log/model.h5")
    print("Saved model to disk")
    # load json and create model
    json_file = open('log/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    return model, X, pd.DataFrame(data = X_standard, columns = X.columns), loaded_model_json

if __name__ == "__main__":
    model, X, X_standard, loaded_model_json = main()
    print("Autoencoder available as model, original DataFrame available as X and normalised DataFrame available as X_standard.")
    