#import mnist_loader
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
'''
print(type(training_data))
for test_3 in training_data:
    print(type(test_3))
    print(type(test_3[0]))
    print(type(test_3[1]))
    print(test_3[0].shape)
    print(test_3[1].shape)
'''    
#import network
#net = network.Network([784, 10, 10])
#net.SGD(training_data, 10, 10, 3.0, test_data=test_data)


import numpy
filename = '../HTRU2/test.csv'
raw_data = open(filename, 'rU')
data = numpy.loadtxt(raw_data, delimiter=',')
training_data = list()
for test in data:
    training_data.append((test[:-1].reshape(8, 1), test[-1].reshape(1, 1)))
print(type(training_data))
for tuple in training_data:
    print(type(tuple))
    print(type(tuple[0]))
    print(tuple[0].shape)
    print(type(tuple[1]))
    print(tuple[1].shape)
import network
net = network.Network([8, 5, 1])
net.SGD(training_data, 2, 2, 3.0, test_data=training_data)
