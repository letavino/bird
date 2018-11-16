import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
import sys
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import keras.backend as K
max_length = 15	
batch_size = 2

def ppd(sen):
	split = sen.split(' ')
	for i in range(len(split), max_length):
		split.append('')
	
	xs = np.repeat(split,batch_size)
	print(xs.shape, split)
	return np.reshape(xs,(batch_size,max_length,1))

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
	
model.fit(x=ppd(" "), y=np.zeros([batch_size,max_length,1]), batch_size=batch_size, epochs=0)

layer_name = 'elmo'

def compare(sen, vecRef):
	vec = model.predict(ppd(sen))
	print(cosine_similarity(np.reshape(vecRef[0,0],(1,1024)), np.reshape(vec[0,0],(1,1024))))

model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
sen = "EU rejects German call to boycott British lamb"
vecRef = model.predict(ppd(sen))
print(len(vecRef[0,0]),np.sum(vecRef[0,0]), vecRef[0,0])

compare("EU - European Union - economic and political union of states mostly located in Europe",vecRef)
compare("EU - europium - chemical element with atomic number of 63",vecRef)
compare("EU - EU - Russian electronic music group",vecRef)
compare("EU - Eastern University - A private university in Bangladesh",vecRef)