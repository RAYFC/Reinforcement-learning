import os
from matplotlib import pyplot as plt
import numpy as np

filename = 'bonus_steps.npy'
filename1 = 'mean.npy'
filename2 = 'stder.npy'
if os.path.exists(filename):
    data1 = np.load(filename)
    data2 = np.load(filename1)
    data3 = np.load(filename2)
    lmda = 0.90
    d = np.mean(data1,axis=0)
    fig,ax = plt.subplots(1)
    ax.plot(range(10))
    fig.text(0.01,0.95,('mean:%s ' %str(data2)))
    fig.text(0.01,0.90,('stder:%s ' %str(data3)))
    plt.plot(np.arange(1, d.shape[0]+1), d, label='Sarsa_lambda={}'.format(lmda))
    plt.ylim([100,500])
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode \naveraged over {} runs'.format(data1.shape[0]))
    plt.legend()
    plt.show()