import matplotlib.pyplot as plt

def plt_params():
    xsmall_size = 12
    small_size = 14
    medium_size = 16
    big_size = 18

    plt.rcParams['font.size'] = small_size 
    plt.rcParams['text.color'] = "white"

    plt.rcParams['axes.facecolor'] = '#001633'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.labelweight'] = 'bold'

    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['xtick.labelsize'] = xsmall_size
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['ytick.labelsize'] = xsmall_size

    plt.rcParams['figure.facecolor'] = '#001633'
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['legend.facecolor'] = 'None'
    plt.rcParams['legend.edgecolor'] = 'None'
    plt.rcParams['legend.fontsize'] = xsmall_size
    plt.rcParams['legend.title_fontsize'] = small_size

    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.transparent'] = False
