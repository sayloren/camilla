'''
Script to call all the others
'''
from read_data import collect_datasets
from make_network import NeuralNetwork
from make_graphs import graph_learning_rate

def get_args():
    parser = argparse.ArgumentParser(description="Description")
    return parser.parse_args()

def evaluate_identity(y,p):
    '''
    '''
    best_case=sum(y==y)
    evaluate=sum(y==p)/best_case
    return evaluate

def main():
    '''
    run the neural net on the training set of pos/neg sequences
    '''
    # read in data from files, positive, negative, test
    # may want to make more specialized negative test sets later
    seq_train,seq_hold,seq_test = collect_datasets(
        'rap1-lieb-positives.txt',
        'yeast-upstream-1k-negative.fa',
        'rap1-lieb-test.txt')

    # set up parameters
    # 17x4 for the bp length of the sequences x the number of
    # elements encoding each
    # one output value
    my_neurons = [68,34,17,9,3,1]
    epochs = 1000
    learning_rate = .001

    # get the columns from the panda into the correct format - np array
    x = np.array(pd.DataFrame(seq_train.binary.values.tolist()))
    y = np.array(pd.DataFrame(seq_train.probability.values.tolist()))
    test = np.array(pd.DataFrame(seq_test.binary.values.tolist()))

    # run on training
    NN = NeuralNetwork(my_neurons)
    error_list,epochs_run = NN.gradient_descent(x,y,epochs,learning_rate)
    print('final error: {0}'.format(error_list[-1]))

    # run on test, make prediction on probabilty of binding, print to
    # file with original sequence
    predictions,_ = NN.feed_forward(test)
    pd_p = pd.DataFrame(predictions[-1])
    pd_p.columns = ['probability']
    pd_predict = pd_p.join(seq_test['sequence'])
    pd_predict.to_csv('probabilities.txt',sep='\t',header=False,index=False)

    # plot
    graph_learning_rate(epochs_run,error_list)

    # cross-validation experiments
    learning_rates = [5,10,25,50,75,100]
    epochs = [10,100,1000]
    hidden = [[68,34,17,9,3,1],[68,68,34,34,17,17,9,9,3,3,1],[68,68,68,34,34,34,17,17,17,9,9,9,3,3,3,1]]
    x = np.array(pd.DataFrame(seq_train.binary.values.tolist()))
    y = np.array(pd.DataFrame(seq_train.probability.values.tolist()))
    test = np.array(pd.DataFrame(seq_test.binary.values.tolist()))

    collect = []
    for l in learning_rates:
        for e in epochs:
            for h in hidden:
                NN = NeuralNetwork(h)
                error_list,epochs_run = NN.gradient_descent(x,y,e,l)
                predictions,_ = NN.feed_forward(x)
                fpr,tpr,_=roc_curve(y,predictions[-1])
                roc_auc=auc(fpr, tpr)
                collect.append([len(h),e,epochs_run,l,error_list[-1],pd_p['probability'].mean(),pd_p['probability'].std(),fpr,tpr])
    pd_params = pd.DataFrame(collect)
    pd_params.columns = ['layer_num','epochs','epochs_run','learning_rate','final_error','ave_prob','std_prob','fpr','tpr']
    pd_params.to_csv('params.txt',sep='\t',header=True,index=False)
    graph_mult_params(pd_params)

if __name__ == "__main__":
    main()
