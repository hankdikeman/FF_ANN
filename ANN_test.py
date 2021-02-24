import numpy as np
import ANN

layer_sizes = (3,4,1,5)
x = np.ones((layer_sizes[0],1))

net = ANN.ANN(layer_sizes)
prediction = net.predict(x)

print(prediction)
