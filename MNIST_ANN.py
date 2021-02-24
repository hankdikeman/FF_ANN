import numpy as np
import matplotlib.pyplot as plt
import os
from ANN import ANN

# read in training data from .npz file
with np.load(os.path.join('data','mnist.npz')) as data:
	training_images = data['training_images']
	training_labels = data['training_labels']

# simple neural net with 1 hidden layer
layer_sizes = (28*28, 32, 10)
# initialize neural net with given layer sizes
net = ANN(layer_sizes)

# train generated model here with SGD

# predict labels of training images
prediction = net.predict(training_images)

# get prediction accuracy of model
accuracy = net.get_classification_accuracy(training_images, training_labels)
print(accuracy)
