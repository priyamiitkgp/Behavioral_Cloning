#!/usr/bin/env python
# coding: utf-8

# In[43]:


import cv2
import csv
from scipy import ndimage
import numpy as np


# In[44]:


lines = []
with open("./data/driving_log.csv") as dl:
    reader = csv.reader(dl)
    for line in reader:
        lines.append(line)


# In[45]:


lines = lines[1:]



corr = 0.2

source_path = "./data/"
images = []
measurements = []
for line in lines:
    m = float(line[3])
    m_list = [m, m+corr, m-corr]
    for i in range(3):
#         try:
        image = cv2.imread(source_path + line[i].strip())
#         print(source_path + line[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1:]
#         image = np.reshape(image, newshape = (image.shape[0], image.shape[1], 1))
        images.append(image)
        measurements.append(m_list[i])
#         except Exception as e:
#             continue


# In[5]:
print(image.shape)

aug_images = []
aug_measure = []


# In[6]:


for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measure.append(measurement)
    aug_images.append(np.fliplr(image))
    aug_measure.append(-1.0*measurement)


# In[7]:


X_train = np.array(aug_images)
y_train = np.array(aug_measure)


# In[19]:


import tensorflow as tf


# In[40]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D


# In[42]:


model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x:x/255.0 - 0.5))
model.add(Conv2D(24, (5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36, (5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48, (5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))


# In[13]:


model.compile(optimizer='adam', loss = 'mse')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs=5)


# In[40]:


model.save('model_hls.h5')

