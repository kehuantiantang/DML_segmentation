# coding=utf-8
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold as manifold
import pickle as pkl
import os
import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def faster_tsne(embedding, labels, class_names):
    label_set = np.unique(labels).tolist()
    fig = plt.figure(figsize=(16,12))
    assert class_names != None, 'You should has class_names while initialize the hook'
    # color = ['blue',  'red', 'purple', 'yellow',
    #          'black', 'green','brown',  'cyan', 'midnightblue', 'orange', 'gray', 'olive'] if len(label_set) <= 12 \
    #     else mcd.CSS4_COLORS.keys()
    cmap = plt.get_cmap('gnuplot')
    color = [cmap(i) for i in np.linspace(0, 1, len(label_set))]

    for i,  c in zip(label_set, color):
        plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], c = c, label=class_names[
            i], s = 10)

    plt.legend(loc='best', prop={'size': 18})
    return fig