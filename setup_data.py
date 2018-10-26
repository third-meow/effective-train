import pickle
import sys
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# where the dataset is
DATA_DIRECTORY = '/home/third-meow/datasets/CMUfaces'
# the categories of images
EXPRESSIONS = ['sad', 'happy', 'angry', 'neutral']  

train_data = []

# 
for expression in EXPRESSIONS:
    # create path to expression
    expression_path = os.path.join(DATA_DIRECTORY, expression)  

    for img_file in os.listdir(expression_path):
        # create path to image
        img_path = os.path.join(expression_path, img_file)

        # load image with opencv
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (70, 70))

        # convert expression into array
        expression_arr = np.zeros(4) 
        if expression == 'angry':
            expression_arr[0] = 1.0

        elif expression == 'happy':
            expression_arr[1] = 1.0
            
        elif expression == 'neutral':
            expression_arr[2] = 1.0

        elif expression == 'sad':
            expression_arr[3] = 1.0


        # append image and label to train data
        train_data.append([img, expression_arr])

# shuffle list of image-label pairs
random.shuffle(train_data)


# split train data into lists x and y for image and label
x = []
y = []
for i in train_data:
    x.append(i[0])
    y.append(i[1])

# pickle the x and y lists
pickle.dump(x, open('saved/xtrain.p', 'wb'))
pickle.dump(y, open('saved/ytrain.p', 'wb'))

