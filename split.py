import numpy as np
import pandas as pd
import sae, read_data

def split_HTRU(x, y):

	np.random.seed(42)

	#Outlier
	index_out = np.where(y == 1)
	x_out = x[index_out]
	y_out = y[index_out]

	#Non-Outlier
	index_in = np.where(y == 0)
	x_in = x[index_in]
	y_in = y[index_in]

	#Train-Test Split
	#Non-Outlier
	index = np.random.choice(16259, 11381, replace = False)
	all_index_train = np.arange(16259)
	index_test = np.setdiff1d(all_index_train, index)
	x_in_train = x_in[index]
	y_in_train = y_in[index]
	x_in_test = x_in[index_test]
	y_in_test = y_in[index_test]

	#Outlier
	index = np.random.choice(1639, 1147, replace = False)
	all_index_train = np.arange(1639)
	index_test = np.setdiff1d(all_index_train, index)
	x_out_train = x_out[index]
	y_out_train = y_out[index]
	x_out_test = x_out[index_test]
	y_out_test = y_out[index_test]

	x_train = np.append(x_in_train, x_out_train, axis = 0)
	y_train = np.append(y_in_train, y_out_train)
	x_test = np.append(x_in_test, x_out_test, axis = 0) #Don't touch this for training now
	y_test = np.append(y_in_test, y_out_test) # And this

	return x_train, y_train, x_test, y_test, x_in_train, y_in_train, x_out_train, y_out_train

def split_credit(x, y):

	np.random.seed(42)

	#Outlier
	index_out = np.where(y == 1)
	x_out = x[index_out]
	y_out = y[index_out]

	#Non-Outlier
	index_in = np.where(y == 0)
	x_in = x[index_in]
	y_in = y[index_in]

	#Train-Test Split
	#Non-Outlier
	index = np.random.choice(284315, 199020, replace = False)
	all_index_train = np.arange(284315)
	index_test = np.setdiff1d(all_index_train, index)
	x_in_train = x_in[index]
	y_in_train = y_in[index]
	x_in_test = x_in[index_test]
	y_in_test = y_in[index_test]

	#Outlier
	index = np.random.choice(492, 344, replace = False)
	all_index_train = np.arange(492)
	index_test = np.setdiff1d(all_index_train, index)
	x_out_train = x_out[index]
	y_out_train = y_out[index]
	x_out_test = x_out[index_test]
	y_out_test = y_out[index_test]

	x_train = np.append(x_in_train, x_out_train, axis = 0)
	y_train = np.append(y_in_train, y_out_train)
	x_test = np.append(x_in_test, x_out_test, axis = 0) #Don't touch this for training now
	y_test = np.append(y_in_test, y_out_test) # And this

	return x_train, y_train, x_test, y_test, x_in_train, y_in_train, x_out_train, y_out_train

def split_PNN(x_in_train, x_out_train, y_in_train, y_out_train):

	num_class = 2
	num_PNN = 10
	k = int(x_in_train.shape[0] / num_PNN)

	x_in_out_train_split = []
	y_in_out_train_split = []

	# Combining outlier class with parts of non-outlier class for each PNN
	for i in range(num_PNN):
		
		if i == num_PNN-1:
			x_in_out_train_split.append(x_in_train[i*k:])
			x_in_out_train_split[i] = np.append(x_in_out_train_split[i], x_out_train, axis = 0)
			y_in_out_train_split.append(y_in_train[i*k:])
			y_in_out_train_split[i] = np.append(y_in_out_train_split[i], y_out_train)

		else:
			x_in_out_train_split.append(x_in_train[i*k:i*k + k])
			x_in_out_train_split[i] = np.append(x_in_out_train_split[i], x_out_train, axis = 0)
			y_in_out_train_split.append(y_in_train[i*k:i*k + k])
			y_in_out_train_split[i] = np.append(y_in_out_train_split[i], y_out_train)

	return x_in_out_train_split, y_in_out_train_split


	