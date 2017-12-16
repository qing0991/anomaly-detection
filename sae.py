import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import read_data

def stackAE1(x):

	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	# Model
	input_layer = Input(shape = (29, ))
	encoded = Dense(5, activation = 'tanh')(input_layer)
	# encoded = Dense(4, activation = 'tanh')(encoded)
	# encoded = Dense(5, activation = 'relu')(encoded)

	# encoded = Dense(1, activation = 'sigmoid')(encoded)

	# decoded = Dense(4, activation = 'tanh')(encoded)
	# decoded = Dense(4, activation = 'tanh')(decoded)
	# decoded = Dense(5, activation = 'relu')(decoded)
	decoded = Dense(29, activation = 'tanh')(encoded)

	autoencoder = Model(input_layer, decoded)
	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')

	autoencoder.fit(x, x,
		epochs = 100,
		batch_size = 256,
		shuffle = True)

	autoencoder.save('Weights/Credit/3.5/sae_credit_ae1')
	return autoencoder

def stackAE2(x):

	# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	# Model
	input_layer = Input(shape = (5, ))
	encoded = Dense(5, activation = 'tanh')(input_layer)
	# encoded = Dense(4, activation = 'tanh')(encoded)
	# encoded = Dense(5, activation = 'relu')(encoded)

	# encoded = Dense(1, activation = 'sigmoid')(encoded)

	# decoded = Dense(4, activation = 'tanh')(encoded)
	# decoded = Dense(4, activation = 'tanh')(decoded)
	# decoded = Dense(5, activation = 'relu')(decoded)
	decoded = Dense(5, activation = 'tanh')(encoded)

	autoencoder = Model(input_layer, decoded)
	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')

	autoencoder.fit(x, x,
		epochs = 100,
		batch_size = 256,
		shuffle = True)

	autoencoder.save('Weights/Credit/3.5/sae_credit_ae2')
	return autoencoder

def stackAE3(x):

	# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	# Model
	input_layer = Input(shape = (5, ))
	encoded = Dense(5, activation = 'tanh')(input_layer)
	# encoded = Dense(4, activation = 'tanh')(encoded)
	# encoded = Dense(5, activation = 'relu')(encoded)

	# encoded = Dense(1, activation = 'sigmoid')(encoded)

	# decoded = Dense(4, activation = 'tanh')(encoded)
	# decoded = Dense(4, activation = 'tanh')(decoded)
	# decoded = Dense(5, activation = 'relu')(decoded)
	decoded = Dense(5, activation = 'tanh')(encoded)

	autoencoder = Model(input_layer, decoded)
	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')

	autoencoder.fit(x, x,
		epochs = 100,
		batch_size = 256,
		shuffle = True)

	autoencoder.save('Weights/Credit/3.5/sae_credit_ae3')
	return autoencoder

# def stackAE4(x):

# 	# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# 	# Model
# 	input_layer = Input(shape = (10, ))
# 	encoded = Dense(10, activation = 'tanh')(input_layer)
# 	# encoded = Dense(4, activation = 'tanh')(encoded)
# 	# encoded = Dense(5, activation = 'relu')(encoded)

# 	# encoded = Dense(1, activation = 'sigmoid')(encoded)

# 	# decoded = Dense(4, activation = 'tanh')(encoded)
# 	# decoded = Dense(4, activation = 'tanh')(decoded)
# 	# decoded = Dense(5, activation = 'relu')(decoded)
# 	decoded = Dense(10, activation = 'tanh')(encoded)

# 	autoencoder = Model(input_layer, decoded)
# 	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')

# 	autoencoder.fit(x, x,
# 		epochs = 100,
# 		batch_size = 256,
# 		shuffle = True)

# 	autoencoder.save('Weights/Credit/3.5/sae_credit_ae4')
# 	return autoencoder

def last(x, y):
	input_layer = Input(shape = (5, ))
	output = Dense(1, activation = 'sigmoid')(input_layer)

	model = Model(input_layer, output)
	model.compile(optimizer = 'adadelta', loss = 'mse')

	model.fit(x, y,
		epochs = 100,
		batch_size = 256,
		shuffle = True)

	model.save('Weights/Credit/3.5/credit_last')
	return model

def total(x, y, x_test, y_test, model):

	model.fit(x, y, 
		epochs = 500,
		batch_size = 256,
		shuffle = True,
		validation_data = (x_test, y_test))

	model.save('Weights/Credit/3.5/credit_total')

	return model

def create_total_model():

	input_layer = Input(shape = (29, ))
	encoded = Dense(5, activation = 'tanh')(input_layer)
	encoded = Dense(5, activation = 'tanh')(encoded)
	encoded = Dense(5, activation = 'tanh')(encoded)
	output = Dense(1, activation = 'sigmoid')(encoded)

	model = Model(input_layer, output)
	model.compile(optimizer = 'adadelta', loss = 'mse', metrics = ['binary_accuracy'])

	return model