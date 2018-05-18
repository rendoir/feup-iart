#Load data
import numpy
import random
import matplotlib.pyplot as plt

def plot_statistics(test_cost, test_accuracy, training_cost, training_accuracy, num_epochs,
               training_cost_xmin=0,
               test_accuracy_xmin=0,
               test_cost_xmin=0,
               training_accuracy_xmin=0,
               training_set_size=14318,
               test_set_size=3580):
    plot_cost(test_cost, training_cost, num_epochs,
                 min(training_cost_xmin, test_cost_xmin))
    plot_accuracy(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size, test_set_size)

def plot_cost(test_cost, training_cost, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(numpy.arange(xmin, num_epochs),
            test_cost[xmin:num_epochs],
            color='#2A6EA6',
            label="Cost on the test data")
    ax.plot(numpy.arange(xmin, num_epochs),
            training_cost[xmin:num_epochs],
            color='#FFA933',
            label="Cost on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()

def plot_accuracy(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size, test_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(numpy.arange(xmin, num_epochs),
            [accuracy*100.0/test_set_size
             for accuracy in test_accuracy],
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(numpy.arange(xmin, num_epochs),
            [accuracy*100.0/training_set_size
             for accuracy in training_accuracy],
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()


numpy.set_printoptions(threshold=numpy.nan)
numpy.seterr(all='ignore')
filename = '../HTRU2/HTRU_2.csv'
file = open(filename, 'rU')
data = numpy.loadtxt(file, delimiter=',')
random.shuffle(data)
#Normalize data
max_values = data.max(axis=0)
min_values = data.min(axis=0)
for i in range(len(data)):
    data[i] = (data[i] - min_values) / (max_values - min_values)
#Statistics
train_positives = 0
test_positives = 0
#Separate training (80%) from testing (20%)
data_split = int(0.8 * len(data))
raw_training_data = data[:data_split]
raw_test_data = data[data_split:]
training_data = list()
for train in raw_training_data:
    train_positives += int(train[8])
    training_data.append((train[:-1].reshape(8, 1), train[-1].reshape(1, 1).astype(int)))
test_data = list()
for test in raw_test_data:
    test_positives += int(test[8])
    test_data.append((test[:-1].reshape(8, 1), test[-1].astype(int)))
print "Train Statistics:"
print "  Positives: {} / {} = {:.2f} %".format(train_positives, len(training_data), 100.0*train_positives/len(training_data))
print "  Negatives: {} / {} = {:.2f} %".format(len(training_data) - train_positives, len(training_data), 100.0*(len(training_data) - train_positives)/len(training_data))
print "Test Statistics:"
print "  Positives: {} / {} = {:.2f} %".format(test_positives, len(test_data), 100.0*test_positives/len(test_data))
print "  Negatives: {} / {} = {:.2f} %\n".format(len(test_data) - test_positives, len(test_data), 100.0*(len(test_data) - test_positives)/len(test_data))

#Run neural network
import network
net = network.Network([8, 100, 1])
test_cost, test_accuracy, training_cost, training_accuracy = net.stochastic_gradient_descent(training_data, 10, 10, 0.1, lmbda = 10.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
n_success, msg = net.accuracy(test_data)
p_accuracy = float(n_success)/len(test_data)*100.0;
print "Accuracy on test data: {} / {} = {:.2f} %\n{}".format(n_success, len(test_data), p_accuracy, msg)
plot_statistics(test_cost, test_accuracy, training_cost, training_accuracy, 10)
