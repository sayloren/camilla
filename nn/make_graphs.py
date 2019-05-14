import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pathlib
import numpy as np
from sklearn.metrics import roc_curve, auc

def graph_learning_rate(epochs_run,error_list_train,error_list_valid,file):
    '''
    make line plots for the train and validation MSE x epochs
    '''
    # set aesthetics
    sns.set_style('ticks')
    sns.set_palette("husl")

    # set mid points for graphs
    half = int(epochs_run/2)

    # set up plot space
    gs = gridspec.GridSpec(3,2,height_ratios=[1,1,1],width_ratios=[1,1])
    gs.update(hspace=.75)
    gs.update(wspace=.5)
    fig = plt.figure(figsize=(10,10))

    # for each subplot (the full epochs and half/half)
    # to look at behavior on different scales

    # first half of epochs
    ax2 = plt.subplot(221)
    h, = ax2.plot(error_list_train[:half])
    j, = ax2.plot(error_list_valid[:half])
    ax2.title.set_text('Epochs 0-{0}'.format(half))

    # second half epochs
    ax3 = plt.subplot(222)
    ax3.plot(error_list_train[half:])
    ax3.plot(error_list_valid[half:])
    ax3.title.set_text('Epochs {0}-{1}'.format(half,epochs_run))

    # all epochs
    ax1 = plt.subplot(223)
    ax1.plot(error_list_train,label='train')
    ax1.plot(error_list_valid,label='valid')
    ax1.title.set_text('All {0} Epochs'.format(epochs_run))

    # legend
    ax4 = plt.subplot(224)
    ax4.axis('off')
    ax4.legend(handles=[h,j],labels=['train','valid'],loc='lower left',bbox_to_anchor=(.000005, .05),fancybox=False, ncol=2)

    # print to folder images
    sns.despine()
    outdir = pathlib.Path('images')
    outfile = outdir / "Error_{0}.png".format(file)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

def graph_vary_params(pd_params_two,file):
    '''
    make graphs of how the parameters vary against the train
    and validation errors
    '''
    sns.set_style('ticks')
    sns.set_palette("husl")

    # set up plot space
    gs = gridspec.GridSpec(2,3,height_ratios=[1,1],width_ratios=[1,1,1])
    gs.update(hspace=.75)
    gs.update(wspace=.5)
    fig = plt.figure(figsize=(10,10))

    # x labels
    error_labels = ['Train','Valid']

    # format data for first subplot
    # learning rate average
    heat_params = pd_params_two[['learning_rate','train_error','valid_error']]
    group_params = heat_params.groupby('learning_rate')['final_error','valid_error'].mean().reset_index()
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax1 = plt.subplot(gs[0,0])
    sns.heatmap(group_params,ax=ax1)
    ax1.title.set_text('Averaged Learning Rate vs MSE')
    ax1.set_xticklabels(error_labels)

    # second subplot - number of layers
    heat_params = pd_params_two[['layer_num','train_error','valid_error']]
    group_params = heat_params.groupby('layer_num')['final_error','valid_error'].mean().reset_index()
    group_params.set_index('layer_num',drop=True,inplace=True)
    ax2 = plt.subplot(gs[0,1])
    sns.heatmap(group_params,ax=ax2)
    ax2.title.set_text('Averaged Number of Layers vs MSE')
    ax2.set_xticklabels(error_labels)

    # third subplot - number of epochs run before convergence
    heat_params = pd_params_two[['epochs_run','train_error','valid_error']]
    group_params = heat_params.groupby('epochs_run')['final_error','valid_error'].mean().reset_index()
    group_params.set_index('epochs_run',drop=True,inplace=True)
    ax3 = plt.subplot(gs[0,2])
    sns.heatmap(group_params,ax=ax3)
    ax3.title.set_text('Averaged Epochs Run vs MSE')
    ax3.set_xticklabels(error_labels)

    # learning rate not averaged
    heat_params = pd_params_two[['learning_rate','train_error','valid_error']]
    heat_params.set_index('learning_rate',drop=True,inplace=True)
    ax4 = plt.subplot(gs[1,0])
    sns.heatmap(heat_params,ax=ax4)
    ax4.title.set_text('Learning Rate vs MSE')
    ax4.set_xticklabels(error_labels)

    # number of layers not averaged
    heat_params = pd_params_two[['layer_num','train_error','valid_error']]
    heat_params.set_index('layer_num',drop=True,inplace=True)
    ax5 = plt.subplot(gs[1,1])
    sns.heatmap(heat_params,ax=ax5)
    ax5.title.set_text('Averaged Number of Layers vs MSE')
    ax5.set_xticklabels(error_labels)

    # epochs run not averaged
    heat_params = pd_params_two[['epochs_run','train_error','valid_error']]
    heat_params.set_index('epochs_run',drop=True,inplace=True)
    ax6 = plt.subplot(gs[1,2])
    sns.heatmap(heat_params,ax=ax6)
    ax6.title.set_text('Epochs Run vs MSE')
    ax6.set_xticklabels(error_labels)

    # print to file
    outdir = pathlib.Path('images')
    outfile = outdir / "Learningrate-error.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

def graph_weight_bias_relation(NN,file):
    '''
    graph how the weights and biases change in the different
    layers (mostly of size 68)
    '''
    # set up plotting params
    sns.set(style="dark")
    f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    # which layer to grabe
    index_nn = [0,1,2,3,4,5,6,7,8]
    for (i,ax,s) in zip(index_nn,axes.flat, np.linspace(0, 3, 10)):
        cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
        tot_w = len(NN.weights[i])*len(NN.weights[i].T)
        x=NN.weights[i].reshape(int(tot_w/2),2)
        y=NN.biases[i].reshape(len(NN.biases[i]),1)
        sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
        ax.title.set_text('Layer: {0}'.format(i))
        ax.set_xlabel('weights')
        ax.set_ylabel('biases')
        ax.set(xlim=(-3, 3), ylim=(-3, 3))

    # print to file
    f.tight_layout()
    outdir = pathlib.Path('images')
    outfile = outdir / "Weights-biases-{0}.png".format(file)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

def graph_roc(fpr,tpr,file,roc_auc):
    '''
    roc auc
    '''
    # plotting params
    sns.set_style('ticks')
    sns.set_palette("husl")

    # plot line
    plt.plot(fpr,tpr)

    # plot labels
    plt.suptitle('Graph of ROC',label='roc')
    plt.title('AUC = {0}'.format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # print to folder images
    sns.despine()
    outdir = pathlib.Path('images')
    outfile = outdir / "ROC_{0}.png".format(file)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

if __name__ == "__main__":
    main()
