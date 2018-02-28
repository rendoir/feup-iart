import sys

'''
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
'''

import numpy
filename = '../HTRU2/HTRU_2.csv'
raw_data = open(filename, 'rU')
data = numpy.loadtxt(raw_data, delimiter=',')

#debug
print(type(data))
print(data.shape)
for row in data: 
    for element in row:
        sys.stdout.write(str(element))
        sys.stdout.write(', ')
    sys.stdout.write('\n')

'''
import network
net = network.Network([784, 10, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
'''