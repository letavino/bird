import matplotlib.pyplot as plt

######################################################
##### Skript zum Plotten des Trainings
######################################################

# Plottet den Verlauf des Trainingsfehlers und des Validierungsfehlers 
# Parameter
#	history: 	HistoryObjekt von Keras fit_generator Funktion

def plotLoss(history):
	# summarize history for loss
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.show()