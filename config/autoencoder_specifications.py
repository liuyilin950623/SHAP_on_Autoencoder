from keras.callbacks import EarlyStopping

dim_hid1 = 8
dim_hid2 = 4
dim_hid3 = dim_hid1
activation = 'relu'
learning_rate = 0.01
batch_size = 128
epochs = 200
early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.01,
                               patience=5)