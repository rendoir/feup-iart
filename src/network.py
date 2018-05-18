import random
import numpy as np

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y)


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        The list ``sizes`` contains the number of neurons in the respective
        layers of the network. The cost is a specific implementation of a generic cost function.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights()
        self.cost=cost

    def init_weights(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, learning_rate,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Train the neural network using batch stochastic gradient
        descent. The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.
        The network's hyper-parameters are:
        ``epochs`` -> Number of training passes
        ``batch_size`` -> Size of each batch
        ``learning_rate`` -> Size of the step in the gradient direction
        ``lmbda`` -> L2 regularization parameter
        The method also accepts ``evaluation_data`` and flags
        that allow us to monitor the cost and accuracy as the network learns.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k+batch_size]
                for k in xrange(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy, msg = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {} = {:.2f} %\n{}".format(accuracy, n, 100.0*accuracy/n, msg)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy, msg = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {} = {:.2f} %\n{}".format(accuracy, n_data, 100.0*accuracy/n_data, msg)
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_batch(self, batch, learning_rate, lmbda, n):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single batch. The ``n``
        parameter is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-learning_rate*(lmbda/n))*w-(learning_rate/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Return the number of inputs in ``data`` for which the neural
        network outputs the correct result.
        The output of binary neural networks is the activation of the output neuron.
        The output of other neural networks is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        if self.sizes[-1] == 1:
            if convert:
                results = [(np.amax(self.feedforward(x)), np.amax(y))
                               for (x, y) in data]
            else:
                results = [(np.amax(self.feedforward(x)), y)
                               for (x, y) in data]
            correct = [0,0]
            total = [0,0]
            for (x, y) in results:
                total[y] += 1
                correct[y] += int(int(round(x)) == y)
            msg = "  Correct positives: {} / {} = {:.2f} %\n  Correct negatives = {} / {} = {:.2f} %".format(
                correct[1],total[1],100.0*correct[1]/total[1],correct[0],total[0],100.0*correct[0]/total[0])
            return correct[0] + correct[1], msg
        else:
            if convert:
                results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                           for (x, y) in data]
            else:
                results = [(np.argmax(self.feedforward(x)), y)
                         for (x, y) in data]
            return sum(int(x == y) for (x, y) in results), ""


    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorize_result(y, self.sizes[-1])
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


#### Miscellaneous functions
def vectorize_result(j, n_out):
    """Creates a vector from the expected result in order to compare to the output."""
    res = np.zeros((n_out, 1))
    if n_out == 1:
        res[0] = j
    else:
        res[j] = 1.0
    return res

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
