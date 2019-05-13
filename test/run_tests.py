from make_network import NeuralNetwork
from read_data import collect_datasets
import numpy as np

def run_auto_encoder():
    '''
    8 3 8 auto encoder
    '''
    # set up parameters
    my_neurons = [8,3,8]

    # both input and output should be identity matrix
    x = np.identity(8)
    y = np.identity(8)
    epochs = 1000
    learning_rate = .0001

    NN = NeuralNetwork(my_neurons)
    error_list,epochs_run = NN.gradient_descent(x,y,epochs,learning_rate)
    prediction,_ = NN.feed_forward(x)

    print('best case is {0}'.format(evaluate_identity(y,y)))
    print('predicted is {0}'.format(evaluate_identity(y,prediction[-1])))

    # check that the matrix indeces are the same
    assert prediction[-1].shape() == y.shape()
    return
