import numpy as np

def scale(x):

	mean_x = np.mean(x, axis = 0)
	std_x = np.std(x, axis = 0)

	x = x - mean_x
	x = x / std_x

	return x