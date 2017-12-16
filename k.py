import numpy as np
import read_data, sae, preprocess, split, gc, pnn, performance_metrics
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import backend as K

def get_output(model, x_train, i):

	get_ith_layer_output = K.function([model.layers[0].input], [model.layers[i].output])
	layer_output = get_ith_layer_output([x_train])[0]
	return layer_output

def greedy_train(x_train):
	
	model = sae.stackAE1(x_train)
	output_ae1 = get_output(model, x_train, 1)

	model = sae.stackAE2(output_ae1)
	output_ae2 = get_output(model, output_ae1, 1)

	model = sae.stackAE3(output_ae2)
	output_ae3 = get_output(model, output_ae2, 1)

	# model = sae.stackAE3(output_ae3)
	# output_ae4 = get_output(model, output_ae3, 1)

	model = sae.last(output_ae3, y_train)
	return model

def extract_weights(model):

	weights = []
	for layer in model.layers:
		weights.append(layer.get_weights())

	return weights

def initialize_weights():
	
	model1 = load_model('Weights/Credit/3.5/sae_credit_ae1')
	model2 = load_model('Weights/Credit/3.5/sae_credit_ae2')
	model3 = load_model('Weights/Credit/3.5/sae_credit_ae3')
	model4 = load_model('Weights/Credit/3.5/credit_last')

	weight = []
	weight.append(extract_weights(model1))
	weight.append(extract_weights(model2))
	weight.append(extract_weights(model3))
	weight.append(extract_weights(model4))

	model = sae.create_total_model()
	i = 0
	for layer in model.layers:
		if i > 0:
			layer.set_weights(weight[i-1][1])
		i += 1

	return model

# def show_weights(model):
	
# 	weights = []
# 	for layer in model.layers:
# 		weights.append(layer.get_weights())
# 		print(layer.get_config())
# 		print()

# 	for i in range(len(weights)):
# 		print((weights[i]))
# 		print()

x, y = read_data.input()
x = preprocess.scale(x)
# x_train, y_train, x_test, y_test, _, _, _, _ = split.split_HTRU(x, y)
x_train, y_train, x_test, y_test, _, _, _, _ = split.split_credit(x, y)

model = greedy_train(x_train)
model = initialize_weights()
model = sae.total(x_train, y_train, x_test, y_test, model)
# model = load_model('Weights/credittotal')

output_to_PNN = get_output(model, x, 2)
output_to_PNN = preprocess.scale(output_to_PNN)
_, _, x_test, y_test, x_in_train, y_in_train, x_out_train, y_out_train = split.split_credit(output_to_PNN, y)
x_in_out_train_split, y_in_out_train_split = split.split_PNN(x_in_train, x_out_train, y_in_train, y_out_train)

num_PNN = 10
num_class = 2

label = pnn.output(x_in_out_train_split, y_in_out_train_split, x_test, y_test, num_PNN, num_class)
label = np.array(label).reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(performance_metrics.accuracy(y_test, label, 2))
print()
print(performance_metrics.confusion_matrix(y_test, label, 2))
print()
print(performance_metrics.precision(y_test, label, 2))
print()
print(performance_metrics.recall(y_test, label, 2))
print()
# show_weights(model)
gc.collect()
