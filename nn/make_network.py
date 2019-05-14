import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# make network structure
class NeuralNetwork(object):
    '''
    initialze the network, pass in a list the length of layers the
    network should have, with each value as the number of nodes in that layer
    the baises and weights are the same dimensions as the network,
    weights are centered at 0 with a spread of .01
    '''    # set random seed
    np.random.seed(10)

    # size argument - scale
    # initiaze the neurons, layers, and their corresponding biases and weights
    def __init__(self,neurons):
        self.neurons = neurons
        self.layers = len(neurons)

        self.biases = [np.random.randn(i) for i in neurons[1:]]
        # matrix for each non-input layer, output x layer dimensions
        self.weights = [np.random.randn(i, j) for i, j in zip(neurons[:-1], neurons[1:])]

    # feedforward
    def feed_forward(self,x):
        '''
        for each pair of weight and biases value, get the dot product of the
        x and the weight, add the bias, then apply the activation function
        save both outputs from both before and after applying the actiavation
        function
        '''
        outputs = []
        products = []

        # for each pair of weight and bias, mult by weight, add bias, then apply actiavation function
        for w,b in zip(self.weights,self.biases):
            # dot product of the weight at that node and the x, adding the bias
            product = np.dot(x, w)+b
            products.append(product)

            # apply activation function, here sigmiod
            activation = sigmoid(product).round(2)
            outputs.append(activation)

            # new x is now the output of the activation
            x = activation
        return outputs,products

    # backpropogate
    def backward_propogation(self,x,y,outputs,products):
        '''
        determining the contribution of the error from each layer
        o - predicted output, y - actual output; get as close to 0 as possible
        find rate of change of loss with respect to weights (derivative)
        gradient descent with mean sum squared loss
        '''
        # initialze the lists for collecting the update values
        weight_deltas = []
        bias_deltas = []

        # use final error to get delta error from actual values
        previous_delta = outputs[-1]-y

        # derivative of the dot product between x and w + b, error contribution
        # for that node
        delta_error_product = previous_delta * sigmoid_derivative(products[-1])

        # collect biases and weights delta error contributions
        bias_deltas.append(delta_error_product)
        weight_deltas.append(np.dot(delta_error_product, outputs[-1].T))

        # extend the x value (essentially the first node/inputs)
        # to all the outputs exept the last in order to have a list of the
        # correct dimension and values to iterate through
        inputs = [x] + outputs[:-1]

        # for each layer, starting from the end, and excluding the first
        # propgate the error contributed
        for i in range(2, self.layers):

            # derivative of the previous product (error contribution of
            # of the previous node; weight*x+bias)
            delta_error = sigmoid_derivative(products[-i])

            # delta error * previous delta
            delta_error_product = np.dot(self.weights[-i+1], delta_error_product) * delta_error

            # append to list of updated weights and biases
            bias_deltas.append(delta_error_product)
            weight_deltas.append(np.dot(delta_error_product, inputs[-i+1]))

        # because working from the end, have to reverse the list now
        reverse_weight = reversed(weight_deltas)
        reverse_bias = reversed(bias_deltas)
        return reverse_weight,reverse_bias

    # train (choose learning method)
    def training_step(self,x,y,learning_rate):
        '''
        each training step first feedfowards, then backpropogates the error
        those values are used to modify the weight and biases in place
        '''
        collect_outputs = []

        # for each pair in input and expected output
        for x_i,y_i in zip(x,y):

            # feedforward
            outputs,products = self.feed_forward(x_i)
            collect_outputs.append(outputs[-1])

            # back prop
            weight_delta,bias_delta = self.backward_propogation(x_i,y_i,outputs,products)

            # update biases and weights
            for (w,b,dw,db) in zip(self.weights,self.biases,weight_delta,bias_delta):
                b-=learning_rate*db
                w-=learning_rate*((dw.T/self.layers)) # normalized
        return collect_outputs

    def gradient_descent(self,x,y,epochs,learning_rate):
        '''
        for the number of epochs, run the training step, each time descending
        further down the gradient
        '''
        # shuffle the indeces of the training data set
        shuffle_index = np.random.permutation(len(x))

        # make cuttoff for test and validation sets
        cutoff = int(len(shuffle_index)*.7)
        train_index = shuffle_index[:cutoff]
        hold_index = shuffle_index[cutoff:]

        # initiate error collection for train and validation
        error_list_train = []
        error_list_valid = []
        previous_out = []

        # the number of elements in the list to check for convergence
        test_len = 1000
        for i in range(epochs):

            # training
            out_train = self.training_step(x[train_index],y[train_index],learning_rate)
            out_valid = self.training_step(x[hold_index],y[hold_index],learning_rate)

            # collect errors
            error_list_valid.append(mean_squared_error(y[hold_index],out_valid))
            error_list_train.append(mean_squared_error(y[train_index],out_train))

            # if there is convergence, leave the loop
            if len(error_list_valid) < test_len:
                pass
            elif sum(error_list_valid[-test_len:])/test_len == error_list_valid[-1]:
                pass
            else:
                print('Converged after {0} epochs'.format(i))
                break
        return error_list_train,error_list_valid,i

def sigmoid(s):
    '''
    sigmoid activation function.
    '''
    return 1/(1+np.exp(np.float64(-s)))

def sigmoid_derivative(s):
    '''
    derivative of sigmoid
    '''
    return sigmoid(np.float64(s)) * (1 - sigmoid(np.float64(s)))

if __name__ == "__main__":
    main()
