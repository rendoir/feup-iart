import numpy
import matplotlib.pyplot as plt

def plot_statistics(test_cost, test_accuracy, training_cost, training_accuracy,
               training_set_size, test_set_size,
               training_cost_xmin=0,
               test_accuracy_xmin=0,
               test_cost_xmin=0,
               training_accuracy_xmin=0):
    num_epochs = len(training_accuracy)
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
