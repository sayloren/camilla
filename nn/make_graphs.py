half = int(epochs/2)
quart = int(half/2)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def get_args():
    parser = argparse.ArgumentParser(description="Description")
    return parser.parse_args()

def main(epochs,error):
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

    # print to folder images
    sns.despine()
    outdir = pathlib.Path('images')
    outfile = outdir / "Error.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile),format='png')
    plt.close()

if __name__ == "__main__":
    main()
