import numpy as np
import json_lines
import time, datetime
from unidecode import unidecode
from keras.utils import to_categorical
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, CuDNNLSTM, Input, concatenate, Bidirectional, Dropout, Lambda
from keras.optimizers import Adagrad

from keras.utils import to_categorical

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Now instantiate the elmo model
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
use_model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# Build our model

# We create a function to integrate the tensorflow model with a Keras model
# This requires explicitly casting the tensor to a string, because of a Keras quirk
def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
	
def UseEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def loadDataset(path, max_length,k=-1, t=None):
	data = list()

	with open(path, 'rb') as f: # opening file in binary(rb) mode    
		for item in json_lines.reader(f):
			if item['gold_label'] == '-':
				item['gold_label'] = 'neutral'
			data.append([item['gold_label'], item['sentence1'], item['sentence2']])

	data = np.array(data[:k])
	
	#print("Text:",data[:,2])
	
	if t == None:
		t = [Tokenizer(),Tokenizer()]
		t[0].fit_on_texts(data[:,0])
	
	encodedLabels = t[0].texts_to_sequences(data[:,0])
	encodedLabels = np.array(encodedLabels).flatten()
	
	padded_sen1 = data[:,1]
	padded_sen2 = data[:,2]

	x_train1 = data[:,1]
	x_train2 = data[:,2]
	y_train = to_categorical(encodedLabels)
	return x_train1, x_train2, y_train, t

max_length = 1 #one sentence instead of 30 words
batch_size = 32
epochs = 300000
steps_per_epoch = 100

x_train1, x_train2, y_train, t = loadDataset(path = "snli_1.0\\snli_1.0_train.jsonl", max_length=max_length)
x_valid1, x_valid2, y_valid, _ = loadDataset(path = "snli_1.0\\snli_1.0_dev.jsonl", max_length=max_length, t=t)

max_features = t[1].document_count

def dataGenerator(x1,x2,y,batch_size):
	import random
	while True:
		start = random.randint(0,len(x1)-batch_size)
		yield ([x1[start:start+batch_size], x2[start:start+batch_size]], y[start:start+batch_size])

split = len(x_train1)*0.8
genTrain = dataGenerator(x_train1,x_train2,y_train,batch_size)
genVal   = dataGenerator(x_valid1,x_valid2,y_valid,batch_size)

print('Build model...')
input1 = Input(shape=(max_length,), dtype=tf.string)
x1 = Lambda(ElmoEmbedding, output_shape=(1024,))(input1)
x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)#dropout=0.2, recurrent_dropout=0.2,
x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)
x1 = Bidirectional(CuDNNLSTM(128, return_sequences=False))(x1)
input2 = Input(shape=(max_length,), dtype=tf.string)
x2 = Lambda(ElmoEmbedding, output_shape=(1024,))(input2)
x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x2)
x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x2)
x2 = Bidirectional(CuDNNLSTM(128, return_sequences=False))(x2)

x3 = Lambda(UseEmbedding, output_shape=(1024,))(input1)
x3 = Dense(256, activation='relu')(x3)

x4 = Lambda(UseEmbedding, output_shape=(1024,))(input2)
x4 = Dense(256, activation='relu')(x4)

x = concatenate([x1,x2,x3,x4])
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#x = Dense(4, activation='relu')(x)
x = Dense(4, activation='softmax')(x)

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
checkpoint = keras.callbacks.ModelCheckpoint(filepath='./models/'+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"), save_best_only=True)

model = Model(inputs=[input1, input2], outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('Train...')
model.fit_generator(generator=genTrain, validation_data=genVal, epochs=epochs, steps_per_epoch = steps_per_epoch, validation_steps=10, callbacks=[tensorboard])# checkpoint
#score, acc = model.evaluate([x_test1, x_test2], y_test, batch_size=batch_size)