# this code is for Keras version 1.2.1
# https://faroit.github.io/keras-docs/1.2.1/
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, PReLU, MaxPooling2D, Dropout, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split

import data_generator as dg
import tensorflow as tf
tf.python.control_flow_ops = tf

# the model implemented here is heavily inspired by 
# https://arxiv.org/pdf/1604.07316.pdf (Nvidia "End to End Learning for Self-Driving Cars")
def get_model():

    model = Sequential()

    # section 14: cropping images in keras
    model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape=(160, 320, 3)))

    # section 9: data preprocessing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Convolutional feature map 24
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2), activation='relu'))
    # http://forums.fast.ai/t/questions-about-batch-normalization/230/3
    model.add(BatchNormalization())

    # Convolutional feature map 36
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    # Convolutional feature map 48
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    # Convolutional feature map 64
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', border_mode='valid', subsample=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    # Convolutional feature map 64
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', border_mode='valid', subsample=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    # fully connected layer
    model.add(Dense(1164, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    # fully connected layer
    model.add(Dense(100, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    # fully connected layer
    model.add(Dense(50, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    # fully connected layer
    model.add(Dense(10, init='he_normal', activation='relu'))

    # output layer
    model.add(Dense(1, init='he_normal'))

    return model

def main():

    # SETUP
    metadata = dg.read_dataset_metadata( "./data/driving_log.csv")
    model = get_model()
    model.summary()

    # generators for training and validation
    train_lines, val_lines = train_test_split(metadata, test_size=0.2)
    train_gen = dg.generator(train_lines, batch_size=128)
    valid_gen = dg.generator(val_lines, batch_size=128)

    # https://faroit.github.io/keras-docs/1.2.1/models/sequential/
    model.compile(optimizer=Adam(1e-3), loss="mse")

    # TRAINING
    # Fits the model on data generated batch-by-batch by a Python generator. 
    history = model.fit_generator(train_gen,
                                samples_per_epoch=20480,
                                nb_epoch=6,
                                validation_data=valid_gen,
                                nb_val_samples=4096,
                                verbose=1)

    # save model
    model.save('model.h5')

# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    main()