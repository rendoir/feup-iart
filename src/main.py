#load data
import numpy
numpy.seterr(all='ignore')
filename = '../HTRU2/HTRU_2.csv'
file = open(filename, 'rU')
data = numpy.loadtxt(file, delimiter=',')
raw_training_data = data[:14318] #training 80%
raw_test_data = data[14318:] #testing 20%
training_data = list()
for test in raw_training_data:
    training_data.append((test[:-1].reshape(8, 1), test[-1].reshape(1, 1).astype(int)))
test_data = list()
for test in raw_test_data:
    test_data.append((test[:-1].reshape(8, 1), test[-1].astype(int)))

#network2.py
import network2
net = network2.Network([8, 100, 1])
net.SGD(training_data, 10, 100, 0.1, lmbda = 10.0,
evaluation_data=test_data,
monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True,
monitor_training_cost=True,
monitor_training_accuracy=True)
print "Accuracy: {0} / {1}".format(net.accuracy(test_data), len(test_data))
