import numpy as np, numpy.random as nr, gym
from itertools import product
import matplotlib.pyplot as plt
import time

def plot_gridworld(mdp,V, pi, s=None, title=None):
    V = V.reshape(mdp.nrow, mdp.ncol)
    plt.figure(figsize=(3,3))
    if title != None:
        plt.title(title)
    plt.imshow(V, cmap='gray')#, clim=(0,1)) 'gist_ncar'
    ax = plt.gca()
    ax.set_xticks(np.arange(V.shape[1])-.5)
    ax.set_yticks(np.arange(V.shape[0])-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:V.shape[0], 0:V.shape[1]]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(0, 1)}
    Pi = pi.reshape(V.shape)
    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
            plt.text(x, y, str(mdp.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    if s != None:
        plt.plot(s%V.shape[0], s//V.shape[0], 'ro')
    plt.grid(color='b', lw=2, ls='-')
    return

def plot_climate(temps, max_temp, title=None):
    plt.figure(figsize=(7,3))
    if title != None:
        plt.title(title)
    temps = list(temps) + [0] * (6 - len(temps))
    temp_mat = np.array([
        [0,0,temps[1],temps[3],temps[3],temps[5],temps[5]],
        [temps[0],temps[0],temps[1],temps[3],temps[3],temps[5],temps[5]],
        [temps[0],temps[0],temps[2],temps[4],temps[4],temps[4],temps[4]]
    ])
    temp_mat = temp_mat / max_temp
    plt.imshow(temp_mat, cmap='inferno', clim=(0,1)) # 'gist_ncar'
    labels = [(0,1,1),(2,0,2),(2,2,3),(3,0,4),(4,2,5),(5,0,6)]
    for x,y,label in labels:
        plt.text(x, y, str(label), color='g', size=12,  verticalalignment='center',
                 horizontalalignment='center', fontweight='bold')
    return

# TODO: need to include value returns for all algorithms
def visualize_grid_history(game,history):
    I = len(game.mdps)
    T = len(history['states'])
    for t in range(T):
        V = history['values'][t]
        pol = history['policies'][t]
        s = history['states'][t]
        ws = ' '.join(["B{}: {}".format(i+1, round(history['beliefs'][t][i],3)) for i in range(I)])
        plot_gridworld(game.true_mdp, V, pol, s, title=ws)

def visualize_climate_history(game,history):
    I = len(game.mdps)
    T = len(history['states'])
    history['actions'] += [None]
    for t in range(T):
        rew = ' '.join(["R{}: {}".format(i+1,history['rewards'][t][i]) for i in range(I)])
        ws = ' '.join(["B{}: {}".format(i+1, round(history['beliefs'][t][i],3)) for i in range(I)])
        s, a = history['states'][t], history['actions'][t-1]
        plot_climate(game.true_mdp.s_to_temps(s), game.true_mdp.nT - 1, title= rew + ' ' + ws + ' act: ' + str(a))