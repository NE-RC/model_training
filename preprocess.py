import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os
import random
import cv2

def reshape_image(path, save_path, new_size = (180, 135)):
    img = cv2.imread(path)
    resized = cv2.resize(img, new_size)
    cv2.imwrite(save_path, resized)

def pickle_data(frames_dir, labels_dir, output_dir, img_size_x, img_size_y):
    training_data = []
    for img in os.listdir(frames_dir):
        i = img[:len(img) - 4]
        img_array = cv2.imread(os.path.join(frames_dir, img))

        f = open(os.path.join(label_dir, str(i) + ".txt"), "r")
        label = float(f.read())
        f.close()

        training_data.append([img_array, label])

        if len(training_data) % 100 == 0:
            print(len(training_data))

    random.shuffle(training_data)

    X = [], Y = []
    for features, label in training_data:
        X.append(features)
        Y.append(label)

    print(len(X))
    print(len(Y))

    X = np.array(X).reshape(-1, img_size_y, img_size_x, 3)
    X = X/255.0

    pickle_out = open(output_dir + "X.pickle", "wb")
    pickle.dump(X, pickle_out, protocol=4)
    pickle_out.close()
    print("Done pickling X!")

    pickle_out = open(output_dir + "Y.pickle", "wb")
    pickle.dump(Y, pickle_out, protocol=4)
    pickle_out.close()
    print("Done pickling Y!")

if __name__ == "__main__":

    #resize the images
    old_dir = "/content/drive/My Drive/TrainingData/newData/frames/"
    new_size = (180, 135)
    new_dir = "/content/drive/My Drive/TrainingData/newData/preprocessed_frames/"
    for img in os.listdir(old_dir):
        img_path = old_dir + img
        new_path = new_dir + img
        reshape_image(img_path, new_path)
    print("Done resizing!")

    base_dir = "/content/drive/My Drive/TrainingData/newData/"
    frames_dir = base_dir + "preprocessed_frames/"
    labels_dir = base_dir + "labels/"

    pickle_data(frames_dir, labels_dir, base_dir, new_size[0], new_size[1])
