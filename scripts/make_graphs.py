half = int(epochs/2)
quart = int(half/2)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


def graph_learning_rate(epochs,error):
    '''
    make line plots for the learning rate x epochs
    '''
    # set aesthetics
    sns.set_style('ticks')
    sns.set_palette("husl")

    # set mid points for graphs
    half = int(epochs/2)
    quart = int(half/2)

    # set up plot space
    gs = gridspec.GridSpec(3,2,height_ratios=[1,1,1],width_ratios=[1,1])
    gs.update(hspace=.75)
    gs.update(wspace=.5)
    fig = plt.figure(figsize=(10,10))

    # subplots
    ax1 = plt.subplot(gs[0,0])
    ax1.plot(error)
    ax1.title.set_text('All {0} Epochs'.format(epochs))

    ax2 = plt.subplot(gs[0,1])
    ax2.plot(error[:quart])
    ax2.title.set_text('Epochs 0-{0}'.format(quart))

    ax3 = plt.subplot(gs[1,0])
    ax3.plot(error[quart:half])
    ax3.title.set_text('Epochs {0}-{1}'.format(quart,half))

    ax4 = plt.subplot(gs[1,1])
    ax4.plot(error[half:half+quart])
    ax4.title.set_text('Epochs {0}-{1}'.format(half,half+quart))

    ax5 = plt.subplot(gs[2,:])
    ax5.plot(error[half+quart:])
    ax5.title.set_text('Epochs {0}-{1}'.format(half+quart,epochs))

    # print to folder
    sns.despine()
    outdir = pathlib.Path('images')
    outfile = outdir / "Error.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

def graph_mult_params(pd_params_two):
    sns.set_style('ticks')
    sns.set_palette("husl")

    gs = gridspec.GridSpec(3,3,height_ratios=[1,1,1],width_ratios=[1,1,1])
    gs.update(hspace=.75)
    gs.update(wspace=.5)
    fig = plt.figure(figsize=(10,10))

    # Grouped Layers
    # totals - learning rate x final error
    heat_params = pd_params_two[['learning_rate','final_error']]
    group_params = heat_params.groupby('learning_rate')['final_error'].mean().reset_index()
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax1 = plt.subplot(gs[0,0])
    sns.heatmap(group_params,ax=ax1)
    ax1.title.set_text('Grouped Learning Rate x Error')

    # totals - learning rate x layers
    heat_params = pd_params_two[['learning_rate','layer_num']]
    group_params = heat_params.groupby('learning_rate')['layer_num'].mean().reset_index()
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax2 = plt.subplot(gs[0,1])
    sns.heatmap(group_params,ax=ax2)
    ax2.title.set_text('Grouped Learning Rate x Number Layers')

    # totals - learning rate x auc
    heat_params = pd_params_two[['learning_rate','auc']]
    group_params = heat_params.groupby('learning_rate')['auc'].mean().reset_index()
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax3 = plt.subplot(gs[0,2])
    sns.heatmap(group_params,ax=ax3)
    ax3.title.set_text('Grouped Learning Rate x AUC')

    # Learning Rate = 5
    # totals - learning rate x final error
    heat_params = pd_params_two[['learning_rate','final_error']]
    group_params = heat_params[heat_params['learning_rate']==5]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax1 = plt.subplot(gs[1,0])
    sns.heatmap(group_params,ax=ax1)
    ax1.title.set_text('Learning Rate = 5 x Error')

    # totals - learning rate x layers
    heat_params = pd_params_two[['learning_rate','layer_num']]
    group_params = heat_params[heat_params['learning_rate']==5]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax2 = plt.subplot(gs[1,1])
    sns.heatmap(group_params,ax=ax2)
    ax2.title.set_text('Learning Rate = 5 x Number Layers')

    # totals - learning rate x auc
    heat_params = pd_params_two[['learning_rate','auc']]
    group_params = heat_params[heat_params['learning_rate']==5]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax3 = plt.subplot(gs[1,2])
    sns.heatmap(group_params,ax=ax3)
    ax3.title.set_text('Learning Rate = 5 x AUC')

    # Learning Rate = 1
    # totals - learning rate x final error
    heat_params = pd_params_two[['learning_rate','final_error']]
    group_params = heat_params[heat_params['learning_rate']==1]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax1 = plt.subplot(gs[2,0])
    sns.heatmap(group_params,ax=ax1)
    ax1.title.set_text('Learning Rate = 1 x Error')

    # totals - learning rate x layers
    heat_params = pd_params_two[['learning_rate','layer_num']]
    group_params = heat_params[heat_params['learning_rate']==1]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax2 = plt.subplot(gs[2,1])
    sns.heatmap(group_params,ax=ax2)
    ax2.title.set_text('Learning Rate = 1 x Number Layers')

    # totals - learning rate x auc
    heat_params = pd_params_two[['learning_rate','auc']]
    group_params = heat_params[heat_params['learning_rate']==1]
    group_params.set_index('learning_rate',drop=True,inplace=True)
    ax3 = plt.subplot(gs[2,2])
    sns.heatmap(group_params,ax=ax3)
    ax3.title.set_text('Learning Rate = 1 x AUC')

    import pathlib
    outdir = pathlib.Path('images')
    outfile = outdir / "Learningrate-error.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()
