import random

import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def extract_features_and_labels(model, dataloader, feat_dim=None, device="cuda"):
    # create empty placeholders
    features = torch.empty((0, feat_dim))
    labels = torch.empty(0, dtype=torch.long)
    for (_, batch) in enumerate(dataloader):
        image = batch['style']
        curr_labels = batch['label']
        curr_feats = model(image.to(device))
        features = torch.cat((features, curr_feats.cpu().detach()))
        labels = torch.cat((labels, curr_labels))
    return features.numpy(), labels.numpy()


def plot_data(tsne, labels, show_c=None):
    classes = np.unique(labels)
    if not show_c:
        show_c = len(classes)
    plt.rcParams["figure.figsize"] = (400, 400)
    # cdict = {1: 'red', 2: 'blue', 3: 'green', 0: 'yellow', 4: 'purple'}
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'gray', 'bisque', 'darkorchid', 'violet', 'slateblue', 'tan',
              'aquamarine', 'greenyellow', 'chocolate']
    # colors = plt.cm.cmaps_listed
    marks = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H',
             '+', 'x', 'X', 'D', 'd', '|', '_', '$m$', '$n$', '$a$', '$f$', '$&$']
    plist = random.shuffle([(c, m) for c in colors for m in marks])
    pdict = {i:v for i, v in enumerate(plist)}
    # first figure for train set features visualization
    fig, ax = plt.subplots()
    for g in random.sample(classes, show_c):
        ix = np.where(labels == g)
        c, m = pdict[g]
        ax.scatter(tsne[ix][:, 0], tsne[ix][:, 1], c=c, marker=m, label=g, s=100)
    ax.legend()
    plt.show()


def show_features(model, dataloader_tr, dataloader_te):
    model.eval()
    # 1.1 Extract train features
    train_features, train_labels = extract_features_and_labels(model, dataloader_tr)

    # 1.2 Extract test features
    test_features, test_labels = extract_features_and_labels(model, dataloader_te)

    tsne = TSNE(n_components=2)
    train_tsne = tsne.fit_transform(train_features)
    test_tsne = tsne.fit_transform(test_features)
    plot_data(train_tsne, train_labels)
    plot_data(test_tsne, test_labels)


colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'gray', 'bisque', 'darkorchid', 'violet', 'slateblue', 'tan',
              'aquamarine', 'greenyellow', 'chocolate']
    # colors = plt.cm.cmaps_listed
marks = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H',
         '+', 'x', 'X', 'D', 'd', '|', '_', '$m$', '$n$', '$a$', '$f$', '$&$']
plist = [(c, m) for c in colors for m in marks]
random.shuffle(plist)
pdict = {i:v for i, v in enumerate(plist)}