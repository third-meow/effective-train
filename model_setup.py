import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten




def add_conv_and_pool(mdl):
    mdl.add(Conv2D(256, (3,3)))
    mdl.add(Activation('relu'))
    mdl.add(MaxPooling2D(pool_size=(2,2)))



def build_model(in_shape):  

    mdl = keras.models.Sequential()

    #conv layers
    mdl.add(Conv2D(256, (3,3), input_shape=in_shape))
    mdl.add(Activation('relu'))
    mdl.add(MaxPooling2D(pool_size=(2,2)))

    add_conv_and_pool(mdl)
    add_conv_and_pool(mdl)

    mdl.add(Flatten())

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(128, activation='sigmoid'))

    mdl.add(Dense(4, activation='sigmoid'))

    mdl.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    return mdl

    
def main():
    # open data
    xtrain = pickle.load(open('saved/xtrain.p', 'rb'))
    ytrain = pickle.load(open('saved/ytrain.p', 'rb'))

    #build and train model
    mdl = build_model(xtrain.shape[1:])
    mdl.fit(xtrain, ytrain, batch_size=16, epochs=5)

    # save model
    mdl.save('saved/latest.h5')

if __name__ == '__main__':
    main()
