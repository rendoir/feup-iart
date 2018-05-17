#Load data
import numpy
import random
numpy.set_printoptions(threshold=numpy.nan)
numpy.seterr(all='ignore')
filename = '../HTRU2/HTRU_2.csv'
file = open(filename, 'rU')
data = numpy.loadtxt(file, delimiter=',')
#Normalize data
max_values = data.max(axis=0)
min_values = data.min(axis=0)
for i in range(len(data)):
    data[i] = (data[i] - min_values) / (max_values - min_values)
#Statistics
train_positives = 0
test_positives = 0
#Separate training (80%) from testing (20%)
raw_training_data = data[:14318]
raw_test_data = data[14318:]
training_data = list()
for train in raw_training_data:
    train_positives += int(train[8])
    training_data.append((train[:-1].reshape(8, 1), train[-1].reshape(1, 1).astype(int)))
test_data = list()
for test in raw_test_data:
    test_positives += int(test[8])
    test_data.append((test[:-1].reshape(8, 1), test[-1].astype(int)))
print "Train Statistics:"
print "  Positives: {0} / {1} = {2:.2f} %".format(train_positives, len(training_data), 100.0*train_positives/len(training_data))
print "  Negatives: {0} / {1} = {2:.2f} %".format(len(training_data) - train_positives, len(training_data), 100.0*(len(training_data) - train_positives)/len(training_data))
print "Test Statistics:"
print "  Positives: {0} / {1} = {2:.2f} %".format(test_positives, len(test_data), 100.0*test_positives/len(test_data))
print "  Negatives: {0} / {1} = {2:.2f} %\n".format(len(test_data) - test_positives, len(test_data), 100.0*(len(test_data) - test_positives)/len(test_data))

#Run neural network
import network2
net = network2.Network([8, 100, 1])
net.SGD(training_data, 10, 100, 0.1, lmbda = 10.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
n_success = net.accuracy(test_data)
p_accuracy = float(n_success)/len(test_data)*100.0;
print "Accuracy: {0} / {1} = {2:.2f} %".format(n_success, len(test_data), p_accuracy)
