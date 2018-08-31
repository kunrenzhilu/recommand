import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

def load_history(path):
    if not 'history.pk' in path:
        path = os.path.join(path, 'history.pk')
    print('load history from {}'.format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def plot_history(dir_list):
    lendir = len(dir_list)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    items = ['accuracy', 'recall', 'precision', 'f1']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    historys = dict()
    for name in dir_list:
        historys[name] = load_history(name)
        
    for item, pos in zip(items, positions):
        legends = []
        for name in dir_list:
            history = historys[name]
            legends.append(axs[pos].plot(history[item], label=name)[0])
        axs[pos].set_title(item)
        axs[pos].legend(handles=legends)
    plt.show()
    return fig

def get_means_and_stds(dir_list, n=10):
    means = defaultdict(list)
    stds = defaultdict(list)
    keys = ['accuracy', 'recall','precision','f1']
    for path in dir_list:
        history = load_history(path)
        for k in keys:
            means[k].append(np.mean(history[k][-n:]))
            stds[k].append(np.std(history[k][-n:]))
    mean = pd.DataFrame(means, columns=keys, index=dir_list)
    std = pd.DataFrame(stds, columns=keys, index=dir_list)
    return mean, std