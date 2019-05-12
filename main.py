import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import glob
import pickle
import numpy as np
import sys, os
import pylab
import time 
import scipy.misc
import matplotlib
import re
from PIL import Image

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Lambda,Input, concatenate, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras.utils import np_utils 
from keras import optimizers 
from keras.models import load_model
from keras.models import model_from_json

import keras.backend as K 
K.set_image_dim_ordering('tf')
print('Image ordering is tf check: ',K.image_dim_ordering())
list_of_losses = []

disease_label_dict = {"Infiltrate":0,"Mass":1,"Nodule":2,"Cardiomegaly":3,"Atelectasis":4,"Effusion":5,"Pneumonia":6,"Pneumothorax":7}
import csv
bbox_path = 'formatted_csv.csv'
csv_as_list = []
with open(bbox_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        csv_as_list.append(row)
csv_as_list = csv_as_list[1:]

#train/test split
train_csv_output = []
train_name = os.listdir("train/")
for information in csv_as_list:
    if information[0] in train_name:
        train_csv_output.append(information)

#train/test split
test_csv_output = []
test_name = os.listdir("test/")
for information in csv_as_list:
    if information[0] in test_name:
        test_csv_output.append(information)

Y = np.zeros((1,16,16))
Y_backup = np.zeros((1,16,16))
Y_label_vectors = []
classification_labels = []
X = []

#Randomizing the images
select_images = np.random.permutation(714)
select_images[0:10]

N_img = 1

for i in select_images: #the random output index
    img_name = train_csv_output[i][0]
    xmin = float(train_csv_output[i][4])
    ymin = float(train_csv_output[i][5])
    xmax = float(train_csv_output[i][6])
    ymax = float(train_csv_output[i][7])
    disease_label = train_csv_output[i][3]
    
    #Loading the image
    img = scipy.misc.imread(os.path.join("train/",img_name))
    
    #print(img.shape)
    
    #Bounding box dimensions 
    avg_x = (xmin+xmax)/2
    avg_y = (ymin+ymax)/2
    bh = ymax - ymin
    bw = xmax - xmin
    
    #Starting pixel values for 64x64 window sliding through the 1024 by 1024 image (16*16)
    nx = 0 
    ny = 0 
    #Moving horizontally through the original image
    
    for yaxis in range(0,16):
        #Starting sliding window from left side of the image 
        nx = 0 
        for xaxis in range(0,16): 
            
            #64 by 64 sub-image 
            window = img[ny:ny+64, nx:nx+64]
            #window starting and ending pixel coordinates 
            startx = nx
            starty = ny 
            endx = nx+63
            endy = ny+63

            #to generate heat map - divide into blocks    
            if avg_x >= startx and avg_x <= endx and avg_y >= starty and avg_y <= endy:
                Y[0,yaxis,xaxis] = 1
            else: 
                Y[0,yaxis,xaxis] = 0
                
            #Moving horizontally
            nx = nx + 64
        #Moving Vertically
        ny = ny + 64

    #Adding Y to the label vector 
    Y_label_vectors.append(np.transpose(Y,axes = [1,2,0]))
    
    #print(Y_label_vectors)
    classification_labels.append(disease_label_dict[disease_label])
    
    #Adding 1024 by 1024 image to the X vector
    X.append(img.reshape(1024,1024,1))

    #Re-initializing Y for the next original image 
    Y = np.zeros((1,16,16))
    Y_backup = np.zeros((1,16,16))
    
    #Keeping track of number of images
    if N_img%100 == 0 : 
        print('Collected '+str(N_img) +' Images and labels')
        
    N_img+=1

#check data generation
import random
X = np.array(X)
print('Shape of X: ',X.shape)
print('Shape of Output Y: ',np.array(Y_label_vectors).shape)

chp = random.choice(range(0,829))
plt.imshow(X[chp].reshape(1024,1024),cmap = 'gray')
plt.show()
plt.imshow(np.array(Y_label_vectors)[chp,:,:,0].reshape(16,16),cmap = 'gray')
plt.show()

X_test =[]
Y = np.zeros((1,16,16))
Y_backup = np.zeros((1,16,16))
Y_label_vectors_test=[]
classification_test=[]

#Randomizing the images
select_images_test = np.random.permutation(129)

N_img = 1

for i in select_images_test: #the random output index
    img_name = test_csv_output[i][0]
    xmin = float(test_csv_output[i][4])
    ymin = float(test_csv_output[i][5])
    xmax = float(test_csv_output[i][6])
    ymax = float(test_csv_output[i][7])
    disease_label = test_csv_output[i][3]
    
    #Loading the image
    img = scipy.misc.imread(os.path.join("test/",img_name))
    #print(img.shape)
    
    #Center of the nodule
    avg_x = (xmin+xmax)/2
    avg_y = (ymin+ymax)/2
    bx = avg_x
    by = avg_y
    
    #Bounding box dimensions 
    avg_x = (xmin+xmax)/2
    abg_y = (ymin+ymax)/2
    bh = ymax - ymin
    bw = xmax - xmin
    
    #Starting pixel values for 64x64 window sliding through the 1024 by 1024 image (16*16)
    nx = 0 
    ny = 0 
    #Moving horizontally through the original image
    
    for yaxis in range(0,16):
        #Starting sliding window from left side of the image 
        nx = 0 
        for xaxis in range(0,16): 
            
            #64 by 64 sub-image 
            window = img[ny:ny+64, nx:nx+64]
            #window starting and ending pixel coordinates 
            startx = nx
            starty = ny 
            endx = nx+63
            endy = ny+63
                
            if avg_x >= startx and avg_x <= endx and avg_y >= starty and avg_y <= endy:
                Y[0,yaxis,xaxis] = 1
            else: 
                Y[0,yaxis,xaxis] = 0
                
                
            #Moving horizontally
            nx = nx + 64
        #Moving Vertically
        ny = ny + 64
    
    #Adding Y to the label vector 
    Y_label_vectors_test.append(np.transpose(Y,axes = [1,2,0]))
    
    #print(Y_label_vectors)
    classification_test.append(disease_label_dict[disease_label])
    
    #Adding 1024 by 1024 image to the X vector
    try:
        X_test.append(img.reshape(1024,1024,1))
    except:
        print(img_name)
        
    #Re-initializing Y for the next original image 
    Y = np.zeros((1,16,16))
    Y_backup = np.zeros((1,16,16))
    
    #Keeping track of number of images
    if N_img%100 == 0 : 
        print('Collected '+str(N_img) +' Images and labels')
        
    N_img+=1

X_test = np.array(X_test)
print('Shape of X: ',X_test.shape)
print('Shape of Output Y: ',np.array(Y_label_vectors_test).shape)

chp = random.choice(range(0,150))
plt.imshow(X_test[chp].reshape(1024,1024),cmap = 'gray')
plt.show()
plt.imshow(np.array(Y_label_vectors_test)[chp,:,:,0].reshape(16,16),cmap = 'gray')
plt.show()

import keras.backend as K
K.clear_session()
input_shape = (1024,1024,1)

model = Sequential([
    Conv2D(32, (3, 3),strides = (1,1), input_shape=input_shape, padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(32, (3, 3),strides = (1,1), padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None) ,
    
    Conv2D(32, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None) ,   
    
    
    Conv2D(64, (3, 3),strides = (1,1), padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(64, (3, 3),strides = (1,1), padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(64, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)  ,
    
    
    
    Conv2D(128, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(128, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(128, (3, 3),strides = (1,1), padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None) ,

    
    Conv2D(256, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
     
    Conv2D(256, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
     
    Conv2D(256, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None) ,

    
    Conv2D(128, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None) ,
    
    Conv2D(64, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),

    
    Conv2D(32, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(16, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(8, (3, 3),strides = (1,1),padding='same',activation='relu'),
    BatchNormalization(axis = -1),
    
    Conv2D(1, (3, 3),strides = (1,1),padding='same',activation='sigmoid',name = 'output'),
    
])

model.summary()


#logistic loss - This loss function is designed purely for object detection. would explore more later
def loss_myconv_entire(y_true,y_pred):
    
    y_true_flat_layer0 = K.flatten(y_true[:,:,:,0])
    y_pred_flat_layer0 = K.flatten(y_pred[:,:,:,0])
    #Logistic loss for the probabilities 
    logistic_loss = K.sum(-y_true_flat_layer0*(K.log(y_pred_flat_layer0)) - (1-y_true_flat_layer0)*(K.log(1-y_pred_flat_layer0)))
    return logistic_loss


#call back function
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_acc =[]
        self.acc = []
        
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs):
        
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
    
    
# Create an instance of the history callback
history_cb = LossHistory()

#Complilation and fitting
opt= optimizers.Adam(lr = 0.001)
batch_size = 8
model.compile(loss=loss_myconv_entire, optimizer=opt)
print('Compiled')

model.fit(np.array(X), np.array(Y_label_vectors), batch_size=batch_size, epochs=15, verbose=1, validation_data=(np.array(X_test), np.array(Y_label_vectors_test)), callbacks = [history_cb])
print('Fitted')
#Save model
model.save('model_15epochs.h5')
#Save model weights
json_string = model.to_json()
model.save_weights('model_heapmap_weights')

with open('model_heatmap.pkl', 'wb') as jm:
    pickle.dump(json_string, jm)

with open('loss_history_heatmap.pkl', 'wb') as lo:
    pickle.dump(history_cb.loss, lo)