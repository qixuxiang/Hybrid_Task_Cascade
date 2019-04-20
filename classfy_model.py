#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:58:17 2019
@author: Mxolisi
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES']='0'

if not os.path.exists('l'):
    os.makedirs('l')

if not os.path.exists('m'):
    os.makedirs('m')

if not os.path.exists('p'):
    os.makedirs('p')

'''
    Here are declarations of variables to be used
'''
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 1
TRAIN_DIR = "/home/airobot/mmdetection/data/jinnan/jinnan2_round2_train_20190401/"
NUM_EPOCHS = 20
'''
    Declarations End
'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize the image to be 224x224 pixels (ignoring
	# aspect ratio), flatten the image into 224x224x3=150528 pixel image
	# into a list, and store the image in the data list
    
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_CUBIC).flatten()
        data.append(image)

    except Exception as e:
        print(str(e))
    
	# extract the class label from the image path and update the
	# labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
'''
data = data.reshape(data.shape[1:])
data = data.transpose()
'''
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

num_train_images = len(trainX)
# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print("len(lb.classes_),",len(lb.classes_))
'''
# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))
# initialize our initial learning rate and # of epochs to train for

INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print(model.summary)
'''
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    base_model = ResNet50(weights='imagenet', 
                      include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


'''
    Here is the code that loads the previous weights and train a new model.
'''

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

class_list = []

for subdir, dirs, files in os.walk(TRAIN_DIR):
    class_name = subdir.split(os.path.sep)[-1]
    if class_name is not TRAIN_DIR.split(os.path.sep)[-1]:
        class_list.append(class_name)



FC_LAYERS = [1024, 1024]
dropout = 0.5
print(class_list)
finetune_model  = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))




adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

filepath = "./checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
shuffle=True, callbacks=callbacks_list)

'''
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=2)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
 



# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

print("Done")
'''