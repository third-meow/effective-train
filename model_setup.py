import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,\
        BatchNormalization


def add_conv_pool(mdl, first=False):
    if first:
        mdl.add(Conv2D(128, (3,3), input_shape=(70, 70, 1)))
    else:
        mdl.add(Conv2D(128, (3,3)))

    mdl.add(Activation('relu'))
    mdl.add(BatchNormalization())
    mdl.add(MaxPooling2D(pool_size=(2,2)))



def build_model():  

    mdl = keras.models.Sequential()
    mdl.add(BatchNormalization())

    # input conv layer
    add_conv_pool(mdl, first=True)

    # hidden conv layers
    add_conv_pool(mdl)
    add_conv_pool(mdl)

    # hidden dense layers
    mdl.add(Flatten())
    mdl.add(Dense(128, activation='softmax'))
    mdl.add(BatchNormalization())
    mdl.add(Dense(128, activation='softmax'))
    mdl.add(BatchNormalization())

    # output layer
    mdl.add(Dense(4, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.001)
    mdl.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return mdl

    
def main():
    # open data
    xtrain = pickle.load(open('saved/xtrain.p', 'rb'))
    ytrain = pickle.load(open('saved/ytrain.p', 'rb'))
    xtest= pickle.load(open('saved/xtest.p', 'rb'))
    ytest= pickle.load(open('saved/ytest.p', 'rb'))

    #build and train model
    mdl = build_model()
    mdl.fit(xtrain, ytrain, batch_size=16, epochs=5)
    print(mdl.evaluate(xtest, ytest))

    # save model
    #mdl.save('saved/latest.h5')

if __name__ == '__main__':
    main()
