import tensorflow as tf # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
from keras import backend as K

import sys, os, datetime, shutil, random, math, hashlib
import numpy as np
import tensorflow_hub as hub
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, CSVLogger, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Input, Lambda, Bidirectional, Dropout, concatenate, PReLU, TimeDistributed, CuDNNLSTM, LSTM, Reshape, Flatten, PReLU, Masking, BatchNormalization, Embedding, subtract
from keras.initializers import he_normal
from keras.constraints import MinMaxNorm	
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from _galLayer import GalDropout
from _candidate import entity, lexem
from _query import getRelScore, getLabel, getInstancesAktiv, getWikiTxt
from instanceComperator import edgeList
from _universalSentenceEncoder import prepareUSE, getVector

from termcolor import cprint
import colorama

### 
# TODO: Trennung von Wiki und Netz
# TO COMBINE: Instance Sequence, Relations

# Vector für text, vector für wiki 
# feature für multi-word einfuegen


import random, sys, os, json
import requests
from _query import getWiki

from _textPreprocessing import text2array, toStringArray, toStringArrayFill
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

			
	sen_len = 256
	wiki_len = sen_len
	inst_len = 25
	inst_vecSize = 16
	rel_len = inst_len
	feat_len = 2

	batch_size = 8
	trainSteps = 100
	valSteps = 100

	def matthews_correlation(self, y_true, y_pred):
		y_pred_pos = K.round(K.clip(y_pred, 0, 1))
		y_pred_neg = 1 - y_pred_pos

		y_pos = K.round(K.clip(y_true, 0, 1))
		y_neg = 1 - y_pos

		tp = K.sum(y_pos * y_pred_pos)
		tn = K.sum(y_neg * y_pred_neg)

		fp = K.sum(y_neg * y_pred_pos)
		fn = K.sum(y_pos * y_pred_neg)

		numerator = (tp * tn - fp * fn)
		denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

		return 1.0 - numerator / (denominator + K.epsilon())
		
	def vecLoss(self, y_true, y_pred):
			
		loss = 0.0
		y_pred = K.permute_dimensions(y_pred,(1,0,2))
		xq = y_pred[0]
		xis = K.permute_dimensions(y_pred[1:],(1,0,2))

		y_true = K.reshape(y_true, (self.batch_size, self.inst_len*2+1))
		for b in range(self.batch_size):
			x = xis[b]-xq[b]
			x = K.sqrt(K.sum(K.pow(x,2), axis=-1))
			v = K.switch(y_true[b][1:], x, 1/(x+K.epsilon()))
			loss += K.mean(v)
		return loss/self.batch_size
	
	def createModel(self, gal=False):
			elmo_model = hub.Module("elmo", trainable=True)
			'''
			module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
			embed = hub.Module(module_url)
			'''
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
			def UniversalEmbedding(x):
				return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
			'''
			'''
			def ElmoEmbeddingI(x):
				tokens_length = np.repeat(self.inst_len,self.batch_size)
				#tokens_length = [self.sen_len,self.sen_len]
				return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tokens_length},signature="tokens",as_dict=True)["elmo"]	
			'''
			inputS 	= Input(shape=(self.sen_len,), dtype='string')
			inputW 	= Input(shape=(self.sen_len,), dtype='string')
			inputI 	= Input(shape=(self.inst_len*2+1, 32,))
			inputR 	= Input(shape=(self.rel_len, 32,))
			inputF 	= Input(shape=(self.feat_len,))
			inputO 	= Input(shape=(self.sen_len,))
			
			xi = inputI
			xr = inputR
			xf = inputF
			xo = inputO
			
			eshape = 1024
			
			eshapeI = 512
			eshapeR = 512
			
			xs = Lambda(ElmoEmbedding, output_shape=(self.sen_len,	eshape))(inputS)
			xw = Lambda(ElmoEmbedding, output_shape=(self.sen_len,	eshape))(inputW)
			#xi = Lambda(ElmoEmbeddingI, output_shape=(self.inst_len,eshapeI))(inputI)
			xo = Reshape([self.sen_len,1])(xo)
			xs = concatenate([xs,xo])
			#xw = concatenate([xs,xo*0.0])
			if self.GPU:
				xs = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=True))(xs)
				xs = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=False))(xs)
				
				xw = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=True))(xw)
				xw = Bidirectional(CuDNNLSTM(units=256, stateful=False, return_sequences=False))(xw)
			else:	
				xs = Bidirectional(LSTM(units=256, stateful=False, return_sequences=True))(xs)
				xs = Bidirectional(LSTM(units=256, stateful=False, return_sequences=False))(xs)
				
				xw = Bidirectional(LSTM(units=256, stateful=False, return_sequences=True))(xw)
				xw = Bidirectional(LSTM(units=256, stateful=False, return_sequences=False))(xw)
	
			#xr = TimeDistributed(Lambda(UniversalEmbedding, output_shape=(1,eshapeR)), name="use_r")(xr)
			'''
			xr = TimeDistributed(Dense(units=128, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			xr = TimeDistributed(Dense(units=64, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			'''
			e2 = Dense(units=32, activation='relu')
			e1 = Dense(units=16, activation='relu')
			e0 = Dense(units=self.inst_vecSize, activation='relu', kernel_constraint=MinMaxNorm(min_value=0.1, max_value=0.9, rate=0.5, axis=0))
			
			
			### CHANGES
			xr = TimeDistributed(Dense(units=4, activation='linear'))(xr)
			xr = TimeDistributed(PReLU())(xr)
			
			xr = Flatten()(xr)
			xr = Dense(units=64, activation='linear')(xr)	
			xr = PReLU()(xr)
			xr = Dense(units=16, activation='linear')(xr)
			xr = PReLU()(xr)
			#xi = TimeDistributed(Lambda(UniversalEmbedding, output_shape=(1,eshapeI)), name="use_i")(xi)
			'''
			xi = TimeDistributed(Dense(units=128, activation='linear'))(xi)
			xi = TimeDistributed(PReLU())(xi)
			xi = TimeDistributed(Dense(units=64, activation='linear'))(xi)
			xi = TimeDistributed(PReLU())(xi)
			'''
			
			xi = TimeDistributed(e2)(xi)
			xi = TimeDistributed(e1)(xi)
			xi = TimeDistributed(e0)(xi)
			xi = Flatten()(xi)
			xio = Reshape((self.inst_len*2+1,self.inst_vecSize), name='embedding')(xi)
			'''
			xi = TimeDistributed(subtract)([xq,xi])
			xi = TimeDistributed(K.pow(xi,2))
			xi = TimeDistributed(K.sum(xi))
			xi = TimeDistributed(K.sqrt(xi))
			xi = Flatten()(xi)
			xi = K.sum(xi)
			'''
			
			'''
			rate = 0.8
			xw = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xw)
			xr = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xr)
			xi = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xi)
			xf = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xf)
			#xo = Dropout(rate=rate, noise_shape=(self.batch_size, 1))(xo)
			'''	
			x = concatenate([xs, xw, xi, xr, xf]) 
					
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
				x = Dense(units=1, activation='sigmoid', name='net')(x)
			
			
			losses = {
				"net": "mse",
				"embedding": self.vecLoss
			}
			metrics = {
				"net": ['acc', self.matthews_correlation],
				"embedding": []
			}
			
			self.model = Model(inputs=[inputS, inputW, inputI, inputR, inputF, inputO], outputs=[x,xio])
			self.model.compile(loss=losses, optimizer='adadelta', metrics=metrics)#binary_crossentropy #nadam # '
			print(self.model.summary())
			return self.model

	def getEncodedSentence():
		
		print("pre. done")
		
		print(v, v.shape)

	def prepareBatch(self, datas):
	
		textArrayArray = []
		cWikisArray = []
		instListArray = []
		relListArray = []
		featuresArray = []
		ohArray = []

		ys = []
		yi = []
		t=10
		dctHex = {
		'0': .0, 
		'1': 1/15/t, 
		'2': 2/15/t, 
		'3': .2/t, 
		'4': 4/15/t, 
		'5': 5/15/t, 
		'6': .4/t, 
		'7': 7/15/t, 
		'8': 8/15/t, 
		'9': 0.6/t, 
		'a': 10/15/t, 
		'b': 11/15/t, 
		'c': .8/t, 
		'd': 13/15/t, 
		'e': (14/15)/t, 
		'f': 1.0/t
		}
	
		for data in datas:
			#hashOfData = str(hash(data.data.tobytes()))
			if False:#hashOfData+".npy" in os.listdir("_cache_prep"):
				hashedData = np.load("_cache_prep/"+hashOfData+".npy")
				[text, cWiki, instList, relList, features, oh, y] = hashedData
				
			else:
				[textArray, index, cid, y] = data
				
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
					a = [dctHex[c] for c in h.lower()]
					relList.append(a)
				for i in range(len(relList),self.rel_len):
					relList.append([.0]*32)
				
				instList = []

				for i in range(min(len(el),self.inst_len)):
					h = hashlib.md5(el[i][0][len('http://www.wikidata.org/entity/Q'):].encode('utf-8')).hexdigest()
					a = [dctHex[c] for c in h.lower()]
					instList.append([a, 1])
					
				if len(instList)==0:
					h = hashlib.md5('none'.encode('utf-8')).hexdigest()
					a = [dctHex[c] for c in h.lower()]
					instList.append([a, 1])
			
				while len(instList) < self.inst_len:
					instList = (instList+instList)[:self.inst_len]
			
				
				#for i in range(len(instList),self.inst_len+1):
				#	instList.append([[.0]*32, 1]) #anpassen, wenn andere hash fkt
				
				for i in range(self.inst_len):
					a = random.choices(list(dctHex.values())[:2],k=32)
					instList.append([a, 0])
			
				#for i in range(len(instList),self.inst_len*2+1):
				#	instList.append([[.0]*32, 0]) #anpassen, wenn andere hash fkt
		
				
				random.shuffle(instList)
				instList 	= np.array(instList)
				
				y2 			= list(instList[:,1])
				instList 	= list(instList[:,0])
				
				h = hashlib.md5(cand.getItem()[len('http://www.wikidata.org/entity/Q'):].encode('utf-8')).hexdigest()
				a = [dctHex[c] for c in h.lower()]
				a = list(np.array(a))
				y2 = [1]+y2
				instList = [a]+instList
				#print(len(y2))
				#print(len(instList))
				'''
				relList = []
				for rel in relA:
					if 'http://www.wikidata.org/entity/Q' in rel['q']['value']:
						nr = rel['q']['value'][len('http://www.wikidata.org/entity/Q'):]
						if os.path.isfile("_cache_wiki/"+nr+".npy"):
							wiki = np.load("_cache_wiki/"+nr+".npy")
						else:
							wiki = getWikiTxt(rel['q']['value'])
							np.save("_cache_wiki/"+nr+".npy",wiki)
						relList.append([wiki])
				for i in range(len(relList),self.rel_len):
					relList.append([''])
					
				instList = []
				for edge in el:
					if 'http://www.wikidata.org/entity/Q' in edge[0]:
						nr = edge[0][len('http://www.wikidata.org/entity/Q'):]
						if os.path.isfile("_cache_wiki/"+nr+".npy"):
							wiki = np.load("_cache_wiki/"+nr+".npy")
						else:
							wiki = getWikiTxt(edge[0])
							np.save("_cache_wiki/"+nr+".npy",wiki)
						instList.append([wiki])
				for i in range(len(instList),self.inst_len):
					instList.append([''])	
				'''
				
				
				text, newIndex 	= toStringArrayFill(textArray, self.sen_len, index=index)
				
				if self.sen_len <= newIndex:
					cprint("Warning prepBatch: index >= sen_len (newIndex: "+str(newIndex)+" >= sen_len: "+str(self.sen_len)+")", "yellow") 
					return None
				
				wikiTxt = cand.getLabel()
				if not cand.getWiki() == '':
					wikiTxt = cand.getWiki()
				elif not cand.getDesc() == '':
					wikiTxt = cand.getDesc()
				
				cWiki = toStringArrayFill(text2array(wikiTxt), self.sen_len)
				'''
				max = 1
				for cd in textArray[index[0]][index[1]].getCandidates():
					if cd.getLinks() > max:
						max = cd.getLinks()
				'''
				#statement = cand.getStatements()
				#cand.getLinks()/max
				features = [cand.getLinks()/10, cand.getNumWords()/2]
				
				oh = [0]*self.sen_len
				oh[newIndex] = 1
				'''
				for i in range(self.sen_len):		
					if not i == newIndex:
						relScore = getRelScore(cand.getItem(), text[i])
						oh[i] = relScore
				'''
				
				
				#x = [text, cWiki, instList[:self.inst_len], features]
				
				#np.save("_cache_prep/"+hashOfData, [text, cWiki, instList, relList, features, oh, y])
			
			textArrayArray.append(text)
			cWikisArray.append(cWiki)
			instListArray.append(instList)
			relListArray.append(relList[:self.rel_len])
			featuresArray.append(features)
			ohArray.append(oh)

			ys.append(y)
			yi.append(y2)
	
		x = [np.array(textArrayArray), np.array(cWikisArray), np.array(instListArray), np.array(relListArray), np.array(featuresArray), np.array(ohArray)]
		
		y = [np.array(ys), np.array(yi)]
		
		return (x,y)
	
	def generator(self, usage):
		#erst alle laden?
		xdata_1			= os.listdir("xdata_1/")
		xdata_1_valid 	= os.listdir("xdata_1_valid/")
		xdata_0 		= os.listdir("xdata_0/")
		xdata_0_valid 	= os.listdir("xdata_0_valid/")
	
		dct = {}
	
	
		'''
		for i in range(len(xdata)):
			dct["xdata/"+xdata[i]] = i+1e2
		for i in range(len(xdata_0)):
			dct["xdata_0/"+xdata_0[i]] = i+1e2
		'''
		files = []
	
		#self.mUSE = prepareUSE(1)
	
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
					data.append(np.load("xdata_1/"+random.choice(xdata_1)))
					
				elif usage=='valid':
					data.append(np.load("xdata_0_valid/"+random.choice(xdata_0_valid)))
					data.append(np.load("xdata_1_valid/"+random.choice(xdata_1_valid)))
					
			batch = self.prepareBatch(data)
			if not batch == None:
				yield batch
			
		textArray = addCandidates2(textArray, maxEntityLength=2, indexIJ=minVal)
		f.close()
		return textArray, minVal
	
	def predict(self, data, runs=1):
		'''
		x,y = self.prepareBatch([data]*self.batch_size)
		preds = []
		for i in range(runs):
			preds.append(self.model.predict_on_batch(x=x))
			print("gal-"+str(i) if runs>1 else "y")
			print(preds[-1], np.mean(preds), np.std(preds),"|\t", y)
		return np.mean(preds), np.std(preds)
		'''
		gen = generator()
		pred = self.model.predict(next(gen))
		
		for p in pred:
			print(p)
		
	def trainDirect(self, epochs, name):
	
		id 			= "XNet_B_"+datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
		tensorboard = TensorBoard(log_dir='./logs/'+id)
		checkpoint 	= ModelCheckpoint(filepath='./models/'+id, save_best_only=True, save_weights_only=True)
		shutil.copyfile(__file__, 'models/'+id+'.py') 
		csv_logger = CSVLogger('C:/Users/Mr_X_/OneDrive/logs/log_'+id+'.log')
		class LossHistory(Callback):
			def __init__(self, model, gen):
				self.model = model
				self.gen = gen
			
			def on_train_begin(self, logs={}):
				self.n = 0

			def on_epoch_end(self, epoch, logs={}):
				#self.losses.append(logs.get('loss'))
				
				if epoch % (10) == 0:
					
					
					#[np.array(textArrayArray), np.array(cWikisArray), np.array(instListArray), np.array(relListArray), np.array(featuresArray), np.array(ohArray)]
					n=0
					with open("batchlog2.txt", 'a', encoding='utf-8') as f:
						for j in range(10):
							x,y = next(self.gen)
							for i in range(len(x[0])):
								pred = self.model.predict(x)
								if pred[i][0] > 0.5 and y[i] <= 0.5 or pred[i][0] <= 0.5 and y[i] > 0.5:
									for word in x[0][i]:
										f.write(word+" ")
									f.write("\n\n")
									for word in x[1][i]:
										f.write(word+" ")
									f.write("\n\n")
									
									f.write(str(pred[i])+" vs. "+str(y[i]))
									f.write("\n------------------------\n")
									n+=1
						f.write("\n########### Epoch: "+str(epoch)+"   "+str(n)+"/80: "+str(int(n/80*100))+"%    ################\n")
		
		genTrain = self.generator('train')
		genValid = self.generator('valid')
	
		cb = LossHistory(self.model,genValid)
	
		self.model.fit_generator(genTrain, steps_per_epoch=self.trainSteps, epochs=epochs, verbose=1, callbacks=[tensorboard,checkpoint,csv_logger], validation_data=genValid, validation_steps=self.valSteps, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

		
if __name__ == "__main__":
	import sys
	for arg in sys.argv:
		print("arg:", arg)
	
	name = 'net3'
	
	net = Net()
	print("create Model")
	net.createModel(gal=False)
	print("train Model")
	net.trainDirect(epochs=150000000, name=name)
	#net.model.load_weights('models/'+name)
	#net.predFlag = False
	#net.trainFlag = False
	#net.trainModel()