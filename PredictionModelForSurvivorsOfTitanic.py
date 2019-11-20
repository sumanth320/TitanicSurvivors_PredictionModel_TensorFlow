#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
from tensorflow.keras.layers import Activation, Dense
import pandas as pd
# loading the dataset from local folder
dataset = pd.read_csv(r"C:/Users/Sumanth/Desktop/Titanic.csv")
# Split the dataset into Input and Output based on file column values
inp = dataset.iloc[:,0:7]
oup = dataset.iloc[:,7]
# Defining the Model
model = Sequential()
#Input paramenter is based on columns in the dataset
model.add(Dense(35, input_dim = 7, activation = 'relu'))
model.add(Dense(22, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
# compiling the model based on an optimizer and loss function to improve efficiency
model.compile(loss = 'logcosh', optimizer = 'adam', metrics = ['accuracy'])
# Fitting the model on the dataset
model.fit(inp, oup, epochs = 200)
# Evaluating the model
accuracy = model.evaluate(inp, oup, verbose = 1)
print('Accuracy: %.2f' % (accuracy*100))

