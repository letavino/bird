import os
import json
import time
import math
import random
import numpy as np

from keras import callbacks
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, PReLU
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from callbacks import StopTraining
from angle import normAngle
from generator import dataGeneratorTraining
from plotter import plotLoss
from galApproach import GalDropout
from TrackingNetwork import ModelLoader, loadBackgroundImg

######################################################
##### Skript zum Erstellen und Trainieren eines Netzes
#####
######################################################

### Initialisierung
batch_size = 32
n_workers = 16

epochs = 12500000
output = 6

imgFolder = "images\\dragon\\"
imgData = "data\\dragon.json"

modelFolder = "models\\"
logFolder = "log\\"
bkgFolder = "background\\"


### Programm

# Liest Bildgröße und Normierung aus JSON
# Parameter
#	imgData: 	JSON Datei
# Return
#	Bildgröße (w,h) in Pixel, Liste der Normierungen der y-Werte

def readConfig(imgData):
	with open(imgData) as f:
		js = json.load(f)
		img_size = js['resolution']
		norm = js['norm']
	return img_size, norm


# Erstellt einen Datensatz 
# Parameter
#	imgFolder: 	Pfad der Bilder
#	imgData: 	JSON Datei, Ground Thruth
#	usage:		Berücksichtigt nur Einträge mit Attribut usage="usage", bei none werden alle geladen
# Return
#	Liste der Dateinamen der Bilder, Liste des Ground Truth	
	
def createDataset(imgFolder, imgData, usage=None):
	with open(imgData) as f:
		js = json.load(f)
		
		norm = js['norm']

		x = []
		y = []

		for pos in js['position']: 
			
			entry = [
				pos['x']/norm[0],
				pos['y']/norm[1],
				pos['z']/norm[2],
				normAngle(pos['relpitch'])/norm[3],
				normAngle(pos['relyaw'])/norm[4],
				normAngle(pos['roll'])/norm[5]
			]
			if usage==None or usage==pos['usage']:
				x.append(pos['name'])
				y.append(entry)
	
	return x, y

img_size, norm = readConfig(imgData)	
input_shape = (img_size[0],img_size[1],3,)
xTrainFiles, yTrainAll = createDataset(imgFolder, imgData, usage='train')
xValidFiles, yValidAll = createDataset(imgFolder, imgData, usage='valid')

print("Build model")
input = Input(shape=input_shape)

baseModel = InceptionResNetV2(include_top=False, weights=None, input_tensor=input, input_shape=input_shape, pooling='max', classes=1000)
baseModel.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = GalDropout(0.25)(baseModel.output)
x = Dense(1000, activation='linear', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Dense(1000, activation='linear', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Dense(100, activation='linear', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Dense(10, activation='linear', kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = PReLU()(x)
x = Dense(output, activation='linear', kernel_initializer='he_normal')(x)

model = Model(inputs = input, outputs = x)

loader = ModelLoader(path="", batch_size=32, norm=norm, r=norm[0]/3, size=img_size)
model.compile(optimizer='Adadelta', loss=loader.proj3d, metrics=[loader.sixDof, loader.sixDofMet])
	
model.summary()

print("Load background images")
bkg1 = loadBackgroundImg(bkgFolder=bkgFolder, num=8000, end=8000)	
bkg2 = loadBackgroundImg(bkgFolder=bkgFolder, num=442, start=8000)	

print("Create generator")	
generatorTrain = dataGeneratorTraining(xTrainFiles, yTrainAll, batch_size, bkg1, imgFolder)
generatorValid = dataGeneratorTraining(xValidFiles, yValidAll, batch_size, bkg2, imgFolder)

print("Create callbacks")
stopCallback = StopTraining()
checkpoint = callbacks.ModelCheckpoint(modelFolder+"weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, period=10)
tbCallback = callbacks.TensorBoard(log_dir=logFolder+str(random.random()), histogram_freq=0, write_graph=True)

### START TRAINING
print("Start training")
history = model.fit_generator(
	generator=generatorTrain,
	steps_per_epoch=len(yTrainAll)/batch_size/1000, 	
	epochs=epochs, 
	verbose=1,
	validation_data=generatorValid,
	validation_steps=len(yValidAll)/batch_size/1000,
	callbacks=[tbCallback,stopCallback,checkpoint],
	workers = n_workers)
				
print("\nTraining done")
model.save('model.h5')
print("\nModel saved")

plotLoss(history.history)
