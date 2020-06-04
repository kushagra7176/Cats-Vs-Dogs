#import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
from tqdm import tqdm
import csv


# Path to test images folder
PATH = r'C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\test1'

labelbin= r'C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\mlb.pickle'
plot= r'C:\Users\kusha\Documents\KERAS\MULTICLASS\keras-multi-label\plot1.png'


# load the trained convolutional neural network and the multi-label
print("[INFO] loading network...")
model = load_model('cats_dogs.model')
mlb = pickle.loads(open(labelbin, "rb").read())


# Sot the images before reading
CAT_classification = []
DOG_classification = []
image_number = []
filenumbers = []

filenames = []
for path in os.listdir(PATH):
    filenumbers.append(int(os.path.splitext(path)[0]))

filenumbers.sort()
print(filenumbers)
for i in filenumbers:
    filenames.append(str(i)+'.jpg')
print("filenames",filenames)


for imageName in tqdm(filenames):
    
    image= os.path.join(PATH,imageName) 
    im_num = os.path.splitext(imageName)[0]
    image_number.append(im_num)
    
    
    # load the image
    image = cv2.imread(image)
    output = imutils.resize(image, width=400)
     
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    
    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    
    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
     	# build the label and draw the label on the image
     	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
     	cv2.putText(output, label, (10, (i * 30) + 25), 
    		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
        if (label == "cat" ):CAT_classification.append(p * 100)
        else:DOG_classification.append(p * 100)
            
    # show the output image
    cv2.imshow("Output", output)
    if cv2.waitKey(0)==27:
        cv2.destroyAllWindows
        
        
cv2.destroyAllWindows

# Save the output to .csv file.

        
with open('FinalOutput.csv', 'w', newline ='') as file:
    writer = csv.writer(file)
    writer.writerow(["SN", "Image Name", "Dog", "Cat", "Binary Classification"])
    for i in range(len(image_number)):
        if DOG_classification[i] > CAT_classification[i]:
            bin_class = 1
        else:
            bin_class = 0
        writer.writerow([i+1, image_number[i], DOG_classification[i], CAT_classification[i],bin_class ])
    
    
    
    