import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os
import cv2

def reshape_image(path, save_path, new_size = (180, 135)):
    img = cv2.imread(path)
    resized = cv2.resize(img, new_size)
    cv2.imwrite(save_path, resized)

old_dir = "/content/drive/My Drive/TrainingData/newData/frames/"
new_size = (180, 135)
new_dir = "/content/drive/My Drive/TrainingData/newData/preprocessed_frames/"
for img in os.listdir(old_dir):
    img_path = old_dir + img
    new_path = new_dir + img
    reshape_image(img_path, new_path)
print("DONE")
