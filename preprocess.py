import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os, cv2, random

IMG_SIZE_X, IMG_SIZE_Y = 180, 135

training_data_dir = "/media/nvidia/F867-A38E/"

old_dir = training_data_dir + "frames/"
new_dir = training_data_dir + "preprocessed-frames/"
labels_dir = training_data_dir + "labels/" 

training_data = []

def reshape_image(path, save_path, new_size = (180, 135)):
    img = cv2.imread(path)
    resized = cv2.resize(img, new_size)
    cv2.imwrite(save_path, resized)


def create_training_data():
    for img in os.listdir(new_dir):
        i = img[:len(img) - 4]
        img_array = cv2.imread(os.path.join(new_dir, img))

        f = open(os.path.join(labels_dir, str(i) + '.txt'), 'r')

        label = float(f.readline())
        f.close()

        training_data.append([img_array, label])

        if len(training_data) % 100 == 0:
            print(len(training_data))

    random.shuffle(training_data)


print("Starting Preprocessing")
print("Loading Images")

#Scale down images and add them to the preprocessed-frames folder

new_size = (IMG_SIZE_X, IMG_SIZE_Y)
for img in os.listdir(old_dir):    
    #print(img)
    img_path = old_dir + img
    new_path = new_dir + img
    reshape_image(img_path, new_path)

print("Finished scaling images down.")
print("Creating training data")

#Create training data

create_training_data()
X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)

assert(len(X) == len(y))

#Remove bad images
for i in range(len(X)):
    if len(X[i]) != IMG_SIZE_Y:
        print("Bad image at index", i)
        X.pop(i)
        y.pop(i)

X = np.array(X).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 3)
X = X/255.0

print("Flipping and appending data")
#flip images and turn angle

X_rev, y_rev = [], []

for x in X:
    X_rev.append(np.fliplr(x))
for orig_label in y:
    y_rev.append(-1.0 * orig_label)
X_rev = np.array(X_rev)
y_rev = np.array(y_rev)

X_total = np.concatenate((X, X_rev))
y_total = np.concatenate((y, y_rev))

np.save(training_data_dir + "X.npy", X_total)
np.save(training_data_dir + "y.npy", y_total)
