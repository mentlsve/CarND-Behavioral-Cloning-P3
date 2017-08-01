# this code is for Keras version 1.2.1
# https://faroit.github.io/keras-docs/1.2.1/
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, PReLU, MaxPooling2D, Dropout
from keras.layers.core import Activation

# https://faroit.github.io/keras-docs/1.2.1/layers/convolutional/
from keras.layers.convolutional import Convolution2D
# https://faroit.github.io/keras-docs/1.2.1/layers/normalization/
from keras.layers.normalization import BatchNormalization


# the model implemented here follows the NVIDIA PIPELINE
def get_model():

    model = Sequential()

    # section 9: data preprocessing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # Convolutional feature map 24
    # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    # ceil(31 - 5 + 1 / 2) = ceil (13.5) = 14
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2)))
    model.add(Activation('tanh'))
    # http://forums.fast.ai/t/questions-about-batch-normalization/230/3
    model.add(BatchNormalization())

    # Convolutional feature map 36
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())

    # Convolutional feature map 48
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, init='glorot_uniform', border_mode='valid', subsample=(2, 2)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())

    # Convolutional feature map 64
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', border_mode='valid', subsample=(1, 1)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())

    # Convolutional feature map 64
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', border_mode='valid', subsample=(1, 1)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())

    model.add(Flatten())

    # fully connected layers
    model.add(Dense(1164, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    # fully connected layers
    model.add(Dense(100, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    # fully connected layers
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    # fully connected layers
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('tanh'))

    # fully connected layers
    model.add(Dense(1, init='he_normal'))
    model.add(Activation('tanh'))

    return model