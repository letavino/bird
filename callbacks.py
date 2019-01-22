import os
from keras import callbacks

class StopTraining(callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if os.path.isfile('./stop'):
			print("Stopped by stop file in epoch",epoch)
			self.model.stop_training = True
			os.rename('./stop','./stop#')

	