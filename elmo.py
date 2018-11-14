import numpy as np
import matplotlib.pyplot as plt
import datetime

from keras.layers import Input, Dense, CuDNNLSTM, Bidirectional, concatenate, Dropout, PReLU, BatchNormalization, Softmax, add, average,Conv1D, MaxPooling2D, Flatten, Reshape, RepeatVector, Conv2D, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import sys

import tensorflow as tf
import tensorflow_hub as hub

def loadData(url):
	xw = []
	yw = []

	xs = []
	ys = []

	maxw = 128

	with open(url) as f:
		for line in f:
			sf = line.split(" ")
			if len(sf)==1:
				xs.append(xw)
				ys.append(yw)
				'''
				if maxw < len(xw):
					maxw = len(xw)
				'''
				xw = []
				yw = []
						
			if len(sf)==4:
				xw.append(sf[0])
				
				if sf[3] == "O\n":
					yw.append(0)
				elif sf[3] == "B-LOC\n":
					yw.append(1)
				elif sf[3] == "I-LOC\n":
					yw.append(2)
				elif sf[3] == "B-PER\n":
					yw.append(3)
				elif sf[3] == "I-PER\n":
					yw.append(4)
				elif sf[3] == "B-ORG\n":
					yw.append(5)
				elif sf[3] == "I-ORG\n":
					yw.append(6)
				elif sf[3] == "B-MISC\n":
					yw.append(7)
				elif sf[3] == "I-MISC\n":
					yw.append(8)
				else:
					print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", sf[3])
				
				
	print(len(xs), maxw)

	for i in range(len(xs)):
		for j in range(maxw-len(xs[i])):
			xs[i].append(' ')
			ys[i].append(0)
	
	xs = np.reshape(xs,(len(xs),128,1))
	ys = to_categorical(ys)
	return np.array(xs),np.array(ys)

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Now instantiate the elmo model
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# Build our model

# We create a function to integrate the tensorflow model with a Keras model
# This requires explicitly casting the tensor to a string, because of a Keras quirk
def ElmoEmbedding(x):
	tokens_length = np.repeat(128,32)
    #return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
	return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tokens_length},signature="tokens",as_dict=True)["elmo"]
	
xTrain, yTrain = loadData("Conll2003\\train.txt")
xValid, yValid = loadData("Conll2003\\valid.txt")
print(xTrain.shape, yTrain.shape)
batch_size = 32
epochs = 0
steps_per_epoch = 100
max_length = 128

def dataGenerator(x,y,batch_size):
	import random
	while True:
		start = 0
		yield (x[start:start+batch_size], y[start:start+batch_size])

genTrain = dataGenerator(xTrain,yTrain,batch_size)
genValid = dataGenerator(xValid,yValid,batch_size)

input = Input(shape=(max_length,1,), dtype=tf.string)
x = Lambda(ElmoEmbedding, output_shape=(max_length,128,), name='elmo')(input)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True, stateful=False))(x)
x = Dense(9, activation='linear')(x)
x = Softmax()(x)

model = Model(inputs=input, outputs=x)
model.compile(loss=categorical_crossentropy, optimizer='adagrad', metrics=['accuracy'], sample_weight_mode="temporal")
print(model.summary())
	

tensorboard = TensorBoard(log_dir='./logs/'+datetime.now().strftime("%y-%m-%d-%H-%M-%S"))

weights = np.ones([128,9])
weights[:,0] = 0.03

model.fit_generator(
	generator=genTrain, 
	validation_data=genValid, 
	epochs=epochs, 
	steps_per_epoch = steps_per_epoch, 
	validation_steps=10, 
	class_weight=weights,
	callbacks=[tensorboard])

data = np.repeat("x", 128)
data[0] = "Hello"
data[1] = "world"
layer_name = 'elmo'
data = next(genTrain)[0]
print(data[0,0])
data[0,0] = "A"
data[0,1] = "seat"
data[0,2] = "is"
data[0,3] = "a"
data[0,4] = "car"
data[0,5] = "from"
data[0,6] = "spain"
data[0,7:128] = ""
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
print(len(intermediate_output[0,0]),np.sum(intermediate_output[0,1]), intermediate_output[0,1])

data[0,0] = "A"
data[0,1] = "seat"
data[0,2] = "is"
data[0,3] = "a"
data[0,4] = "place"
data[0,5] = "to"
data[0,6] = "sit"
data[0,7:128] = ""
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
print(len(intermediate_output[0,0]),np.sum(intermediate_output[0,1]), intermediate_output[0,1])