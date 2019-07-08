import tensorflow as tf # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
from keras import backend as K

import sys, os, datetime, shutil, random, math, hashlib
import numpy as np
import tensorflow_hub as hub
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, CSVLogger, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Input, Lambda, Bidirectional, Dropout, concatenate, PReLU, TimeDistributed, CuDNNLSTM, LSTM, Reshape, Flatten, PReLU, Masking, BatchNormalization, Embedding
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from _galLayer import GalDropout
from _candidate import entity, lexem
from _query import getRelScore, getLabel, getInstancesAktiv
from instanceComperator import edgeList

from termcolor import cprint
import colorama

### 
# TODO: Trennung von Wiki und Netz
# TO COMBINE: Instance Sequence, Relations

import random, sys, os, json
import requests
from _query import getWiki

from _textPreprocessing import text2array, toStringArray
from create import addCandidates2
			
class Net():

	

	num_cores = 4
	GPU = True
	print("GPU:", GPU)

	CPU = not GPU
	if GPU:
		num_GPU = 1
		num_CPU = 1
	if CPU:
		num_GPU = 0
		num_CPU = 1
		
	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
			inter_op_parallelism_threads = num_cores, allow_soft_placement=True,\
			device_count = {'CPU': num_CPU, 'GPU': num_GPU})
	session = tf.Session(config=config)
	K.set_session(session)

	seedVal = 3
	random.seed(seedVal)
	np.random.seed(seedVal)
	tf.set_random_seed(seedVal)
	
	
	colorama.init()

			
	sen_len = 512
	wiki_len = sen_len
	inst_len = 25
	rel_len = inst_len
	feat_len = 2

	batch_size = 2
	trainSteps = 120
	valSteps = 90

	GPU = True
	trainFlag = True
	predFlag = True
	preFlag = False
		
	def createModel(self, gal=False):
			elmo_model = hub.Module("elmo", trainable=True)
			K.set_session(self.session)
			self.session.run(tf.global_variables_initializer())
			self.session.run(tf.tables_initializer())

			'''
			word_emb: the character-based word representations with shape [batch_size, max_length, 512].
			lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
			lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
			elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
			default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
			'''

			def ElmoEmbedding(x):
				tokens_length = np.repeat(self.sen_len,self.batch_size)
				#tokens_length = [self.sen_len,self.sen_len]
				return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tokens_length},signature="tokens",as_dict=True)["elmo"]	
			'''
			def ElmoEmbeddingI(x):
				tokens_length = np.repeat(self.inst_len,self.batch_size)
				#tokens_length = [self.sen_len,self.sen_len]
				return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tokens_length},signature="tokens",as_dict=True)["elmo"]	
			'''
			inputS 	= Input(shape=(self.sen_len,), dtype='string')
			inputW 	= Input(shape=(self.sen_len,), dtype='string')
			inputI 	= Input(shape=(self.inst_len,32,))
			inputR 	= Input(shape=(self.rel_len,32,))
			inputF 	= Input(shape=(self.feat_len,))
			inputO 	= Input(shape=(self.sen_len,))
			
			xi = inputI
			xr = inputR
			xf = inputF
			xo = inputO
			
			eshape = 1024
			eshapeI = 1024
			xs = Lambda(ElmoEmbedding, output_shape=(self.sen_len,	eshape))(inputS)
			xw = Lambda(ElmoEmbedding, output_shape=(self.sen_len,	eshape))(inputW)
			#xi = Lambda(ElmoEmbeddingI, output_shape=(self.inst_len,eshapeI))(inputI)
			#xr = Lambda(ElmoEmbeddingI, output_shape=(self.inst_len,eshapeI))(inputR)
			
			xs = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=True))(xs)
			xs = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=False))(xs)
			
			xw = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=True))(xw)
			xw = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=False))(xw)
			'''
			xi = Conv1D(128, (3), padding="same", activation="relu")(xi)
			xi = MaxPooling1D(pool_size=(2))(xi)
			xi = Conv1D(64, (3), padding="same", activation="relu")(xi)
			xi = Flatten()(xi)
			'''
			'''
			xi = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=True))(xi)
			xi = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=False))(xi)
			'''
			'''
			xi = Bidirectional(CuDNNLSTM(units=32, stateful=False, return_sequences=True))(xi)
			xi = Bidirectional(CuDNNLSTM(units=1, stateful=False, return_sequences=True))(xi)
			xi = Flatten()(xi)
			'''
			'''
			xr = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=True))(xr)
			xr = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=False))(xr)
			'''
			xi = TimeDistributed(Dense(units=16, activation='linear'))(xi)
			xi = TimeDistributed(PReLU())(xi)
			xi = TimeDistributed(Dense(units=4, activation='linear'))(xi)
			xi = TimeDistributed(PReLU())(xi)
			xi = TimeDistributed(Dense(units=1, activation='linear'))(xi)
			xi = TimeDistributed(PReLU())(xi)
			xi = Flatten()(xi)
			
			xr = TimeDistributed(Dense(units=16, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			xr = TimeDistributed(Dense(units=4, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			xr = TimeDistributed(Dense(units=1, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			xr = Flatten()(xr)
			
			'''
			xr = Bidirectional(CuDNNLSTM(units=32, stateful=False, return_sequences=True))(xr)
			xr = Bidirectional(CuDNNLSTM(units=1, stateful=False, return_sequences=True))(xr)
			xr = Flatten()(xr)
			'''
			#xi = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=True))(xi)
			
			#xi = Bidirectional(CuDNNLSTM(units=64, stateful=False, return_sequences=False))(xi)
			
			
			
			'''
			rate = 0.15
			xw = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xw)
			xr = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xr)
			xi = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xi)
			xf = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xf)
			#xo = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xo)
			'''		
			x = concatenate([xs, xw, xi, xf, xo]) 
							
			#x = Dropout(0.4)(x)
			if gal:
				x = Dense(units=1024, activation='linear')(x)
				x = PReLU()(x)
				x = GalDropout(0.5)(x)
				x = Dense(units=256, activation='linear')(x)
				x = PReLU()(x)
				x = GalDropout(0.5)(x)
				x = Dense(units=64, activation='linear')(x)
				x = PReLU()(x)
				x = GalDropout(0.4)(x)
				x = Dense(units=16, activation='linear')(x)
				x = PReLU()(x)
				x = GalDropout(0.4)(x)
				x = Dense(units=1, activation='sigmoid')(x)
			else:
				x = Dense(units=1024, activation='linear')(x)
				x = PReLU()(x)
				x = Dropout(0.5)(x)
				x = Dense(units=256, activation='linear')(x)
				x = PReLU()(x)
				x = Dropout(0.5)(x)
				x = Dense(units=64, activation='linear')(x)
				x = PReLU()(x)
				x = Dropout(0.4)(x)
				x = Dense(units=16, activation='linear')(x)
				x = PReLU()(x)
				x = Dropout(0.4)(x)
				x = Dense(units=1, activation='sigmoid')(x)
			
			
			self.model = Model(inputs=[inputS, inputW, inputI, inputR, inputF, inputO], outputs=x)
			self.model.compile(loss='mse', optimizer='adadelta', metrics=['acc'])#binary_crossentropy #nadam
			print(self.model.summary())
			return self.model

	def prepareBatch(self, datas):
	
		textArrayArray = []
		cWikisArray = []
		instListArray = []
		relListArray = []
		featuresArray = []
		ohArray = []
		ys = []
	
		for data in datas:
			[textArray, index, cid, y] = data
			
			'''
			if len(textArray)-1 < index[0]:
				print(">>", y, textArray[0][0].getWord(), textArray[0][1].getWord())
			elif len(textArray[index[0]])-1 < index[1]:
				print(">", y, textArray[0][0].getWord(), textArray[0][1].getWord())
			elif len(textArray[index[0]][index[1]].getCandidates())-1 < cid:
				print("\n\n>>>", y, textArray[0][0].getWord(), textArray[0][1].getWord(),"\n\n")
			'''
			cand = textArray[index[0]][index[1]].getCandidates()[cid] 
			el = edgeList(cand.getItem(), 'P31', entity=cand, blacklistFlag=False)
			nr = cand.getItem()[len('http://www.wikidata.org/entity/Q'):]
			if os.path.isfile("rels/"+nr+".npy"):
				relA = np.load("rels/"+nr+".npy")
			else:
				relA = getInstancesAktiv(cand.getItem())
				np.save("rels/"+nr, relA)
			
			relList = []
			for rel in relA:
				h = hashlib.md5(rel['l']['value'][len('http://www.wikidata.org/entity/Q'):].encode('utf-8')).hexdigest()
				a = [ord(c)-47 for c in h.upper()]
				relList.append(a)
			for i in range(len(relList),self.rel_len):
				relList.append([0]*32)
			
			instList = []
			for edge in el:
				h = hashlib.md5(edge[0][len('http://www.wikidata.org/entity/Q'):].encode('utf-8')).hexdigest()
				a = [ord(c)-47 for c in h.upper()]
				instList.append(a)
			for i in range(len(instList),self.inst_len):
				instList.append([0]*32) #anpassen, wenn andere hash fkt
			
			text, newIndex 	= toStringArray(textArray, self.sen_len, index=index)
			
			if self.sen_len <= newIndex:
				cprint("Warning prepBatch: index >= sen_len (newIndex: "+str(newIndex)+" >= sen_len: "+str(self.sen_len)+")", "yellow") 
				return None
			
			cWiki 	= toStringArray(text2array(cand.getWiki()), self.sen_len)
			
			max = 1
			for cd in textArray[index[0]][index[1]].getCandidates():
				if cd.getLinks() > max:
					max = cd.getLinks()
			
			#statement = cand.getStatements()
			
			features = [cand.getLinks(), cand.getLinks()/max]
			
			oh = [0]*self.sen_len
			oh[newIndex] = 1
			'''
			for i in range(self.sen_len):		
				if not i == newIndex:
					relScore = getRelScore(cand.getItem(), text[i])
					oh[i] = relScore
			'''
			
			
			#x = [text, cWiki, instList[:self.inst_len], features]
			
			
			
			textArrayArray.append(text)
			cWikisArray.append(cWiki)
			instListArray.append(instList[:self.inst_len])
			relListArray.append(relList[:self.rel_len])
			featuresArray.append(features)
			ohArray.append(oh)
			ys.append(y)
			'''
			print("text:", text)
			print("wiki:", cWiki)
			print("inst:", instList[:self.inst_len])
			print("feat:", features)
			print("oh:", oh)
			print("y:", y)
			'''
		x = [np.array(textArrayArray), np.array(cWikisArray), np.array(instListArray), np.array(relListArray), np.array(featuresArray), np.array(ohArray)]
		
		y = np.array(ys)
		
		return (x,y)
		return (x,y)

	def generatorOld(self, usage):
		#erst alle laden?
		xdata 			= os.listdir("xdata/")
		xdata_valid 	= os.listdir("xdata_valid/")
		xdata_0 		= os.listdir("xdata_0/")
		xdata_0_valid 	= os.listdir("xdata_0_valid/")
	
		while True:
		
			data = []
			
			halfBatchSize = int(self.batch_size/2)	

			for i in range(halfBatchSize):
				if usage=='train':	
					data.append(np.load("xdata_0/"+random.choice(xdata_0)))
					data.append(np.load("xdata/"+random.choice(xdata)))
				elif usage=='valid':
					data.append(np.load("xdata_0_valid/"+random.choice(xdata_0_valid)))
					data.append(np.load("xdata_valid/"+random.choice(xdata_valid)))
					
			batch = self.prepareBatch(data)
			if not batch == None:
				yield batch
	
	
	def generator(self, usage, cb):
		#erst alle laden?
		xdata 			= os.listdir("xdata/")[:621]
		xdata_valid 	= os.listdir("xdata_valid/")
		xdata_0 		= os.listdir("xdata_0/")[:621]
		xdata_0_valid 	= os.listdir("xdata_0_valid/")
	
		dct = {}
	
	
	
		for i in range(len(xdata)):
			dct["xdata/"+xdata[i]] = i+1e2
		for i in range(len(xdata_0)):
			dct["xdata_0/"+xdata_0[i]] = i+1e2
	
		files = []
	
		while True:
			#print("test:", cb.test) #im ersten Schritt sinnlos
			'''
			for file in files:
				dct[file] = cb.test #so: komplette uebernahme des Werts, vllt besser Schnittbildung
			
				
			sortedDct = sorted(((value, key) for (key,value) in dct.items()), reverse=True) #nicht alles sortieren, sondern einordnen
			
			valInvDecay = 0.01 # increase 0.01 per epoche
			for key,value in dct.items():
				dct[key] += valInvDecay/(self.batch_size*self.trainSteps)
			'''
			data = []
			files = []
			halfBatchSize = int(self.batch_size/2)	

			for i in range(halfBatchSize):
				if usage=='train':	
					'''
					files.append(sortedDct[ min(abs(int(random.gauss(0,.33*len(sortedDct)))),len(sortedDct)-1) ][1])	#nicht nur von unten, sondern auch mal von mitte, oben -> loss == wahrscheinlichkeit der Ziehung, immer mal wieder alle durchgehen
					files.append(sortedDct[ min(abs(int(random.gauss(0,.33*len(sortedDct)))),len(sortedDct)-1) ][1])
						
					#print(files[-2])
					#print(files[-1])
						
					data.append(np.load(files[-2]))	
					data.append(np.load(files[-1]))
					'''
					data.append(np.load("xdata_0/"+random.choice(xdata_0)))
					data.append(np.load("xdata/"+random.choice(xdata)))
					
				elif usage=='valid':
					data.append(np.load("xdata_0_valid/"+random.choice(xdata_0_valid)))
					data.append(np.load("xdata_valid/"+random.choice(xdata_valid)))
					
			batch = self.prepareBatch(data)
			if not batch == None:
				yield batch
	
	def fkt2(self,txt): #besser 'wsl', statt 'min'
		f = open("dict.json","r")
		st = f.read()
		dct = json.loads(st)

		maxDct = max(dct.keys(), key=(lambda k: dct[k]))
		
		textArray = text2array(txt)
		
		#ri = random.randrange(len(textArray))
		#rj = random.randrange(len(textArray[ri]))
		min = 1E6
		minVal = []
		
		for i in range(len(textArray)):
			for j in range(len(textArray[i])):
				
				w = textArray[i][j].getWord().lower()
				if w in dct:
					if dct[w] < min:
						min = dct[w]
						minVal = [i,j]
				else:
					min = 1
					minVal = [i,j]
					dct[w] = 1
					#save dct
			
		textArray = addCandidates2(textArray, maxEntityLength=2, indexIJ=minVal)
		f.close()
		return textArray, minVal
	
	def predict(self, data, runs=1):
		
		x,y = self.prepareBatch([data,data])
		preds = []
		for i in range(runs):
			preds.append(self.model.predict_on_batch(x=x))
			print("gal-"+str(i) if runs>1 else "y")
			print(preds[-1], np.mean(preds), np.std(preds))
		return np.mean(preds), np.std(preds)
		
	def trainDirect(self, epochs, name):
	
		id 			= "XNet_"+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
		tensorboard = TensorBoard(log_dir='./logs/'+id)
		checkpoint 	= ModelCheckpoint(filepath='./models/'+name, save_best_only=True, save_weights_only=True)
		shutil.copyfile(__file__, 'models/'+id+'.py') 
		
		class LossHistory(Callback):
			def on_train_begin(self, logs={}):
				self.test = 1e2
				

			def on_batch_end(self, batch, logs={}):
				#self.losses.append(logs.get('loss'))
				self.test = logs.get('loss')
			
		cb = LossHistory()
		
		genTrain = self.generator('train', cb)
		genValid = self.generator('valid', cb)
		#
		self.model.fit_generator(genTrain, steps_per_epoch=self.trainSteps, epochs=epochs, verbose=1, callbacks=[tensorboard,checkpoint, cb], validation_data=genValid, validation_steps=self.valSteps, class_weight=None, max_queue_size=6, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

		
if __name__ == "__main__":
	import sys
	for arg in sys.argv:
		print("arg:", arg)
	
	name = 'net3'
	
	net = Net()
	print("create Model")
	net.createModel(gal=False)
	print("train Model")
	net.trainDirect(epochs=400000, name=name)
	#net.model.load_weights('models/'+name)
	#net.predFlag = False
	#net.trainFlag = False
	#net.trainModel()
