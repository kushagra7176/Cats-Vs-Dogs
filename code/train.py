

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

#************************************************************************************************************************************#
#************************************************************************************************************************************#
######                                                    MODEL TRAINING                                                        ######  
#************************************************************************************************************************************#
#************************************************************************************************************************************#

# Initialize the Hyperparameters: --epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# Det the PATH to the Dataset, pickle file to store label_binarizer values
dataset=r'C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\train'
labelbin= r'C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\mlb.pickle'


# Randomply shuffle the images:
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)


######################################################################################################################################
######                                                       Data Pre-processing                                                ######  
######################################################################################################################################

# initialize the data and labels
data = []
labels = []


# Load all input images
for imagePath in imagePaths:
    
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    
    # Convert image to numpy array
	image = img_to_array(image)
	data.append(image)

	# Extract set of class labels from the image path and update the labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# Standardize the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# Binarize the labels using scikit-learn's multi-labelbinarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


######################################################################################################################################
######                                                      TRAINING THE MODEL                                                  ######  
######################################################################################################################################


print("[INFO] compiling model...")


import time
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
# from tensorflow.keras.callbacks import TensorBoard
# import time
NAME = "cats&Dogs_{}".format( int(time.time()))
        
print(NAME)
tensorboard = TensorBoard(log_dir=r"C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\logs\{}".format(NAME))


# model = SmallerVGGNet.build(
# 	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
# 	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
# 	)

width=IMAGE_DIMS[1]
height=IMAGE_DIMS[0]
depth=IMAGE_DIMS[2]
classes=len(mlb.classes_)

model = Sequential()
      
 

inputShape = (height, width, depth)
chanDim = -1

# if we are using "channels first", update the input shape
# and channels dimension
if K.image_data_format() == "channels_first":
	inputShape = (depth, height, width)
	chanDim = 1

# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same",
	input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
   

# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[tensorboard])

# save the model to disk
print("[INFO] serializing network...")
model.save('cats_dogs.model')



# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(labelbin, "wb")
f.write(pickle.dumps(mlb))
f.close()

