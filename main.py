import cv2 
import os 
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize


ROOT_DIR = '/Users/safwankhan/Desktop/Projects/Deeplearning/Cancer/data/'
healthy_imgs = os.listdir(ROOT_DIR + 'Healthy/')
tumor_imgs = os.listdir(ROOT_DIR + 'Tumor/')
dataset = []
label = []

# Healthy 
for _,img_name in enumerate(healthy_imgs):
    if(img_name.split('.')[1]== 'jpg'):
        image = cv2.imread(ROOT_DIR + 'Healthy/' + img_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

# Tumor
for _,img_name in enumerate(tumor_imgs):
    if(img_name.split('.')[1]== 'jpg'):
        image = cv2.imread(ROOT_DIR + 'Tumor/' + img_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

# Convert dataset & label into array using numpy
dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize()