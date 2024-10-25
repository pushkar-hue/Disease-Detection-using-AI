import cv2 as cv
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import os

normaldir = 'TB_Chest_Radiography_Database/Normal'
tbdir = 'TB_Chest_Radiography_Database/Tuberculosis'
images = []
labels = []
imagesize = 256

for x in os.listdir(normaldir):
    imagedir = os.path.join(normaldir, x)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(0)
    
for y in os.listdir(tbdir):
    imagedir = os.path.join(tbdir, y)
    image = cv.imread(imagedir, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (imagesize, imagesize))
    images.append(image)
    labels.append(1)
    
    
#Converting to NumPy arrays since they have more features than regular lists
images = np.array(images)
labels = np.array(labels)

#Splitting the images and labels into training and testing sets, then normalizing the values within them for computational efficiency (from 0-255 scale to 0-1 scale)
imagetrain, imagetest, labeltrain, labeltest = train_test_split(images, labels, test_size=0.3, random_state=42)
imagetrain = (imagetrain.astype('float32'))/255
imagetest = (imagetest.astype('float32'))/255


#Flattening the image array into 2D (making it [2940 images] x [all the pixels of the image in just one 1D array]) to be suitable for SMOTE oversampling
imagetrain = imagetrain.reshape(2940, (imagesize*imagesize))

#Performing oversampling
smote = SMOTE(random_state=42)
imagetrain, labeltrain = smote.fit_resample(imagetrain, labeltrain)

#Unflattening the images now to use them for convolutional neural network (4914 images of 256x256 size, with 1 color channel (grayscale, as compared to RGB with 3 color channels))
imagetrain = imagetrain.reshape(-1, imagesize, imagesize, 1)
print(imagetrain.shape)

#Classes balanced - equal counts of each label
print(np.unique(labeltrain, return_counts=True))


#The CNN model has 3 convolutional layers, each followed by pooling to summarize the features found by the layer, starting with 16 and multiplying by 2 each time for computational efficiency, as bits are structured in powers of 2. 3x3 filters and ReLU activation used.
cnn = keras.Sequential(
    [
    #Input layer, same shape as all the images (256x256x1):
    keras.Input(shape=(imagesize, imagesize, 1)),
    
    #1st convolutional layer:
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    #2nd convolutional layer:
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    #3rd convolutional layer:
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    #Flattening layer for the dense layers:
    Flatten(),
    
    #1st dense layer following the convolutional layers:
    Dense(64, activation='relu'),
    
    #Dropout layer with heavy dropout rate to avoid overfitting in the large-ish dataset
    Dropout(0.5),
    
    #Output layer that squeezes each image to either 0 or 1 with sigmoid activation
    Dense(1, activation='sigmoid')
    ]
)


#Compiling the model with parameters best suited for the task at hand:
cnn.compile(
    loss='binary_crossentropy', #Best for binary classification
    optimizer = keras.optimizers.Adam(learning_rate=0.001), #Good starting LR for dataset of this size
    metrics=['accuracy'], #Looking for accuracy
)


#Fitting the model, with the ReduceLROnPlateau callback added to it to reduce the learning rate to take smaller steps in increasing the accuracy whenever the learning rate plateaus (goes in the wrong direction)
#Doing this with patience=1, meaning it will perform this if it even plateaus for one epoch, since only 10 epochs are used
#factor=0.1 means that for every time the learning rate is reduced, it is reduced by a factor of 0.1 - it also won't go lower than 0.00001

reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=1, min_lr=0.00001, verbose=1)

#Fitting the model w/ the callback. ON VS CODE, batch size of 16 makes each epoch take around a minute in this case w/ good accuracy, making the whole training process 10 min, but on Kaggle it should take longer due to less computational resources:
cnn.fit(imagetrain, labeltrain, batch_size=16, epochs=10, verbose=2, callbacks = [reduce_lr])



#Evaluating the data w/ multiple types of metrics
print('TESTING DATA:')
cnn.evaluate(imagetest, labeltest, batch_size=32, verbose=2)

print('ADVANCED TESTING METRICS:')
predictions = cnn.predict(imagetest, batch_size=32)
predicted_labels = (predictions > 0.5).astype('int32')
print(classification_report(labeltest, predicted_labels))
print(confusion_matrix(labeltest, predicted_labels))
cnn.save('model.h5')
