'''import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 10, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)'''


import numpy
numpy.seterr(all='ignore')
filename = '../HTRU2/HTRU_2.csv'
file = open(filename, 'rU')
data = numpy.loadtxt(file, delimiter=',')
training_data = list()
for test in data:
    training_data.append((test[:-1].reshape(8, 1), test[-1].reshape(1, 1).astype(int)))
import network
net = network.Network([8, 5, 1])
net.SGD(training_data, 10, 10, 3.0, test_data=None)
