import numpy as np

# simple class for a feedforward neural network
class ANN:
	def __init__(self, layer_size):
		# set weight shapes using given layer sizes
		self.weight_shape = [(a,b) for a,b in zip(layer_size[1:], layer_size[:-1])]
		# initialize weights
		self.weights = [np.random.standard_normal(shape)/np.power(shape[1],0.5) for shape in self.weight_shape]
		# initialize biases
		self.biases = [np.zeros((s,1)) for s in layer_size[1:]]

	# predict output from input matrix
	def predict(self, x):
		for wt,bi in zip(self.weights, self.biases):
			x = self.activation(np.matmul(wt, x) + bi)
		return x

	def get_classification_accuracy(self, inputs, labels):
		predictions = self.predict(inputs)
		correct = np.sum([np.argmax(x) == np.argmax(y) for x,y in zip(predictions, labels)])
		return correct / np.shape(predictions)[0]

	
	# tanh activation function
	@staticmethod
	def activation(x):
		return np.tanh(x)

	def train(self, train_data, epochs, batchsize):
		pass 

	# cross-entropy loss function
	@staticmethod
	def xe_loss(prediction_matrix, training_label):
		return -1 * training_label * np.log(prediction_matrix)
