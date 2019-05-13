# make network structure
class NeuralNetwork(object):
    np.random.seed(10)

    def __init__(self,neurons):
        '''
        initialze the network, pass in a list the length of layers the
        network should have, with each value as the number of nodes in that layer
        the baises and weights are the same dimensions as the network,
        weights are centered at 0 with a spread of .01
        '''
        self.neurons = neurons # number of neurons
        self.layers = len(neurons) # number of layers to make

        # size argument - scale
        self.biases = [np.random.randn(i, 1) for i in neurons[1:]] # column of random values for each non-input layer
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
            activation = sigmoid(product).round(3)
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
        # for each x,y pair
        for i,j in zip(x,y):

            # run feed forward
            outputs,products = self.feed_forward(i)

            # get mean square error for the output of the feed foward and the actual y
            mean_sq_error = mean_squared_error(j,outputs[-1])

            # if the sum of the actual is equal to the predicted, break
            if sum(j) == sum(outputs[-1]):
                break

            # run back prop
            weight_delta,bias_delta = self.backward_propogation(i,j,outputs,products)

            # update the weights and biases
            for (w,b,dw,db) in zip(self.weights,self.biases,weight_delta,bias_delta):
                b-=learning_rate*db
                w-=learning_rate*((dw.T/self.layers)) # normalized
        return mean_sq_error

    def gradient_descent(self,x,y,epochs,learning_rate):
        '''
        for the number of epochs, run the training step, each time descending
        further down the gradient
        '''

        # shuffle the data sets by index
        shuffle_index = np.random.permutation(len(x))

        # make a 70:30 cutoff for training and validation sets
        cutoff = int(len(df_shuffle)*.7)

        # seperate train and hold by index
        train_index = df_shuffle[:cutoff]
        hold_index = df_shuffle[cutoff:]

        # initialize list for collecting error
        error_list = []

        # the length of the list from which to check for convergence
        test_len = 10

        # run the training step for number of epochs and collect error at each
        for i in range(epochs):

            # run on test and hold out set
            error_test = self.training_step(x[train_index],y[train_index],learning_rate)
            error_hold = self.training_step(x[hold_index],y[hold_index],learning_rate)
            error_list.append(error_hold)

            # if the average of the last n elements is the same as the last element, has converged
            if len(error_list) < test_len:
                pass
            elif sum(error_list[-test_len:])/test_len == error_list[-1]:
                pass
            else:
                print('Converged after {0} epochs'.format(i))
                break
        return error_list,i

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
