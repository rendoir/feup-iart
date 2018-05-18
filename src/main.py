import network
import plot

import random
import sys
import numpy as np

#Load data
np.set_printoptions(threshold=np.nan)
np.seterr(all='ignore')
filename = '../HTRU2/HTRU_2.csv'
file = open(filename, 'rU')
data = np.loadtxt(file, delimiter=',')
n_data = len(data)
random.shuffle(data)
p_flag = '-p' in sys.argv or '--plot' in sys.argv
m_flag = '-m' in sys.argv or '--monitor' in sys.argv or p_flag
#Normalize data
max_values = data.max(axis=0)
min_values = data.min(axis=0)
for i in range(n_data):
    data[i] = (data[i] - min_values) / (max_values - min_values)
#Statistics
train_positives = 0
test_positives = 0
#Separate training (80%) from testing (20%)
n_train = int(0.8 * n_data)
n_test = n_data - n_train
raw_training_data = data[:n_train]
raw_test_data = data[n_train:]
training_data = list()
for train in raw_training_data:
    train_positives += int(train[8])
    training_data.append((train[:-1].reshape(8, 1), train[-1].reshape(1, 1).astype(int)))
test_data = list()
for test in raw_test_data:
    test_positives += int(test[8])
    test_data.append((test[:-1].reshape(8, 1), test[-1].astype(int)))
print "Train Statistics:"
print "  Positives: {} / {} = {:.2f} %".format(train_positives, n_train, 100.0*train_positives/n_train)
print "  Negatives: {} / {} = {:.2f} %".format(n_train - train_positives, n_train, 100.0*(n_train - train_positives)/n_train)
print "Test Statistics:"
print "  Positives: {} / {} = {:.2f} %".format(test_positives, n_test, 100.0*test_positives/n_test)
print "  Negatives: {} / {} = {:.2f} %\n".format(n_test - test_positives, n_test, 100.0*(n_test - test_positives)/n_test)

#Run neural network
net = network.Network([8, 8, 8, 1])
test_cost, test_accuracy, training_cost, training_accuracy = net.stochastic_gradient_descent(training_data, 10, 8, 0.25, lmbda = 0.75,
        evaluation_data=test_data,
        monitor_evaluation_cost=m_flag,
        monitor_evaluation_accuracy=m_flag,
        monitor_training_cost=m_flag,
        monitor_training_accuracy=m_flag)
n_success, msg = net.accuracy(test_data)
p_accuracy = float(n_success)/n_test*100.0;
print "Accuracy on test data: {} / {} = {:.2f} %\n{}".format(n_success, n_test, p_accuracy, msg)
if p_flag:
    plot.plot_statistics(test_cost, test_accuracy, training_cost, training_accuracy, n_train, n_test)
