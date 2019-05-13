import numpy as np

import pytest
from nn.make_network import NeuralNetwork
from nn.read_data import collect_datasets

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

def reverse_binary_dictionary(sequence):
    '''
    convert binary sequence back to nucleotide
    '''
    binDict = {'1000':'A','0100':'C','0010':'G','0001':'T'}
    list_str = [str(i) for i in sequence]
    join_str = "".join(list_str)
    parse_str = [join_str[0+i:4+i] for i in range(0, len(join_str), 4)]
    return "".join([binDict[i] for i in parse_str])

def test_sequence_encoding():
    '''
    check that the data set sequences are being read in correctly, that they can
    be converted back
    '''

    # get the data sets
    seq_train,seq_hold,seq_test = collect_datasets(
        'rap1-lieb-positives.txt',
        'yeast-upstream-1k-negative.fa',
        'rap1-lieb-test.txt')

    seq_hold['reverse_binary'] = [reverse_binary_dictionary(seq) for seq in seq_hold['sequence']]
    seq_hold['tf']=seq_hold['sequence']==seq_hold['rev_binary']
    assert seq_hold['tf'].all() == True

if __name__ == "__main__":
    main()
