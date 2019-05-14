'''
Script to call all the others
'''
from .read_data import collect_datasets
from .make_network import NeuralNetwork
from .make_graphs import graph_learning_rate,graph_vary_params,graph_weight_bias_relation,graph_roc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Description")
    parser.add_argument("-e", "--epochs",type=int,help='number of epochs to run - typically 1000')
    parser.add_argument("-l", "--learn",type=float,help='learning rate - typically .01')
    parser.add_argument("-n","--neurons",default="68, 34, 1",nargs='+',help='number of neurons per layer, note: number of layers will be inferred, so you need at least 3')
    return parser.parse_args()

def main():
    '''
    run the neural net on the training set of pos/neg sequences
    '''
    args = get_args()
    # read in data from files, positive, negative, test
    # may want to make more specialized negative test sets later
    seq_train,seq_hold,seq_test = collect_datasets(
        'rap1-lieb-positives.txt',
        'yeast-upstream-1k-negative.fa',
        'rap1-lieb-test.txt')

    # get the columns from the panda into the correct format - np array
    x = np.array(pd.DataFrame(seq_train.binary.values.tolist()))
    y = np.array(pd.DataFrame(seq_train.probability.values.tolist()))

    # if you provide the single parametrs, run as a single run, otherwise
    # run the cross validation set
    if args.learn is not None:

        # set parameters
        best_neurons = args.neurons
        epochs = args.epochs
        best_lr = args.learn
        file_label = 'best_params_3-1k-.01'

        # intialize net
        NN = NeuralNetwork(best_neurons)

        # train
        error_train_list,error_valid_list,epochs_run = NN.gradient_descent(x,y,epochs,l)

        # run on input data to evaluate efficieny
        predictions,_ = NN.feed_forward(x)

        # true and fasle positive rates
        fpr,tpr,_=roc_curve(y,predictions[-1])

        # roc curve
        roc_auc=auc(fpr, tpr)

        # graphing
        graph_learning_rate(epochs_run,error_train_list,error_valid_list,file_label)
        graph_roc(fpr,tpr,file_label,roc_auc)
        graph_weight_bias_relation(NN,file_labels)

        # convert the test data into correct format for running in net
        test = np.array(pd.DataFrame(seq_test.binary.values.tolist()))

        # get predictions
        predictions,_ = NN.feed_forward(test)

        # convert to panda to print out to file
        pd_p = pd.DataFrame(predictions[-1])
        pd_p.columns = ['probability']
        pd_predict = pd_p.join(seq_test['sequence'])
        pd_predict.to_csv('probabilities.txt',sep='\t',header=False,index=False)

    else:
        # cross-validation experiments
        learning_rates = [.001,.01,.1,1]
        epochs = 1000
        hidden = [[68,34,1],[68,34,17,9,3,1],[68,68,68,68,68,34,34,34,34,34,17,17,17,9,9,9,3,3,3,1]]

        # collect params out
        collect = []

        # iterate through parameters
        for l in learning_rates:
            for h in hidden:
                file_labels = '{0}-{1}-{2}'.format(l,epochs,len(h))

                # initialze net
                NN = NeuralNetwork(h)

                # train
                error_train_list,error_valid_list,epochs_run = NN.gradient_descent(x,y,epochs,l)

                # make predictions based on inputs in order to evalue efficiency
                predictions,_ = NN.feed_forward(x)

                # get false and true positive rates
                fpr,tpr,_=roc_curve(y,predictions[-1])

                # calcute roc
                roc_auc=auc(fpr, tpr)

                # graph how the error rates compare in different runs with different parameters
                graph_learning_rate(epochs_run,error_train_list,error_valid_list,file_labels)

                # graph the roc
                graph_roc(fpr,tpr,file_labels,roc_auc)

                # if sufficently many layers, get the distributions of biases and weights in those layers
                if len(h) > 17:
                    graph_weight_bias_relation(NN,file_labels)

                # collect all the parameters
                collect.append([len(h),epochs,epochs_run,l,error_train_list[-1],error_valid_list[-1],fpr,tpr])#,pd_p['probability'].mean(),pd_p['probability'].std()

        # get parameters into dataframe and save to file, then graphic representations
        pd_params = pd.DataFrame(collect)
        pd_params.columns = ['layer_num','epochs','epochs_run','learning_rate','train_error','valid_error','fpr','tpr']#,'ave_prob','std_prob'
        pd_params.to_csv('params.txt',sep='\t',header=True,index=False)
        graph_vary_params(pd_params)

if __name__ == "__main__":
    main()
