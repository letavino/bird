import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
import sys
import tensorflow as tf
import tensorflow_hub as hub

import keras.backend as K

batch_size = 2

def ppd(sen, max_length, toSplit=True):
	if toSplit:
		sen = sen.split(' ')
	for i in range(len(sen), max_length):
		sen.append('')
	
	xs = np.repeat(sen,batch_size)
	#print(xs.shape, split)
	return np.reshape(xs,(batch_size,max_length,1))

def prepareElmo(max_length):
	sess = tf.Session()
	K.set_session(sess)
	elmo_model = hub.Module("elmo", trainable=True)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.tables_initializer())

	def ElmoEmbedding(x):
		tokens_length = np.repeat(max_length,batch_size)
		return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tokens_length},signature="tokens",as_dict=True)["elmo"]
		'''
		
		word_emb: the character-based word representations with shape [batch_size, max_length, 512].
		lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
		lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
		elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
		default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
		'''

	input = Input(shape=(max_length,1,), dtype=tf.string)
	x = Lambda(ElmoEmbedding, output_shape=(max_length,1024,), name='elmo')(input)
	x = Dense(1, activation='linear')(x)

	model = Model(inputs=input, outputs=x)
	model.compile(loss='mse', optimizer='adagrad')
		
	model.fit(x=ppd(" ", max_length), y=np.zeros([batch_size,max_length,1]), batch_size=batch_size, epochs=0)

	layer_name = 'elmo'
	model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
	return model

def compare(sen, model, max_length, vecRef, start, end):
	vec = model.predict(ppd(sen, max_length))
	cos = np.empty(end-start)

	arg1 = np.reshape(np.mean(vecRef[0],axis=0),(1,1024))
	arg2 = np.reshape(np.mean(vec[0],axis=0),(1,1024))
	print("shape:", arg1.shape)
	cos = cosine_similarity(arg1, arg2)
	return cos
	
def comparex(sen, model, max_length, vecRef, start, end):
	vec = model.predict(ppd(sen, max_length))
	cos = np.empty(end-start)
	for i in range(start,end):
		arg1 = np.reshape(vecRef[0,i],(1,1024))
		arg2 = np.reshape(vec[0,0],(1,1024))
		cos[i] = cosine_similarity(arg1, arg2)
	return cos

def getVector(sen, model, max_length):
	vec = model.predict(ppd(sen, max_length,toSplit=False))
	return vec
'''
vecRef, cos = compare("EU rejects German call to boycott British lamb")

vecRef, cos1 = compare("EU - European Union - economic and political union of states mostly located in Europe",vecRef)
vecRef, cos2 = compare("EU - europium - chemical element with atomic number of 63",vecRef)
vecRef, cos3 = compare("EU - EU - Russian electronic music group",vecRef)
vecRef, cos4 = compare("EU - Eastern University - A private university in Bangladesh",vecRef)

print(cos1,cos2,cos3,cos4)
'''