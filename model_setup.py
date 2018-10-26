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



def build_model(shp):  
    '''
    plan:
        conv layer
        conv layer
        conv layer
        dense layer
        dense layer
        dense layer
    '''

    mdl = keras.models.Sequential()

    #conv layers
    mdl.add(Conv2D(256, (3,3), input_shape=shp[1:]))
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
    xtrain = pickle.load(open('saved/xtrain.p', 'rb'))
    xtrain = np.array(xtrain)
    xtrain = xtrain.reshape((-1, 70, 70, 1))

    ytrain = pickle.load(open('saved/ytrain.p', 'rb'))
    ytrain = np.array(ytrain)
    #ytrain = ytrain.reshape((-1, 4, 1))

    print('xtrain is type {} and shape {}'.format(type(xtrain), xtrain.shape))
    print('ytrain is type {} and shape {}'.format(type(ytrain), ytrain.shape))
    mdl = build_model(xtrain.shape)
    mdl.fit(xtrain, ytrain, epochs=1)

    #pickle.dump(open('saved/lastest.model', 'wb'))


if __name__ == '__main__':
    main()
