import random
from collections import OrderedDict

import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

from data import create_dataset, dataset_catalog
from models.StyleEncoder_model import StyleEncoder
from options.train_options import TrainOptions


def extract_features_and_labels(model, dataloader, feat_dim=512, device="cuda"):
    # create empty placeholders
    features = torch.empty((0, feat_dim))
    labels = torch.empty(0, dtype=torch.long).to(device)
    for (i, batch) in enumerate(dataloader):
        image = batch['style']
        curr_labels = batch['label'].to(device)
        curr_feats = model(image.to(device))

        features = torch.cat((features, torch.squeeze(curr_feats).cpu().detach()))
        labels = torch.cat((labels, curr_labels))
    return features.cpu().numpy(), labels.cpu().numpy()


def plot_data(tsne, labels, show_c=None):
    classes = np.unique(labels)
    if not show_c:
        show_c = len(classes)
    plt.rcParams["figure.figsize"] = (15, 15)
    #plt.rcParams['figure.dpi'] = 400
    # cdict = {1: 'red', 2: 'blue', 3: 'green', 0: 'yellow', 4: 'purple'}
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'gray', 'bisque', 'darkorchid', 'violet', 'slateblue', 'tan',
              'aquamarine', 'greenyellow', 'chocolate']
    # colors = plt.cm.cmaps_listed
    marks = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H',
             '+', 'x', 'X', 'D', 'd', '|', '_', '$m$', '$n$', '$a$', '$f$', '$&$']
    plist =[(c, m) for c in colors for m in marks]
    random.shuffle(plist)
    pdict = {i:v for i, v in enumerate(plist)}
    # first figure for train set features visualization
    try:
        fig, ax = plt.subplots()
        for g in classes:
            ix = np.where(labels == g)
            c, m = pdict[g]

            ax.scatter(tsne[ix][:, 0], tsne[ix][:, 1], c=c, marker=m,label=g, s=100)
        #ax.legend()
        fig.savefig(opt.dataname+ str(len(labels))+".png")
        #plt.show()
    except Exception as e:
        print(e)


def show_features(model, dataloader_tr, dataloader_te):
    model.eval()
    tsne = TSNE(n_components=2)

    # 1.1 Extract train features
    train_features, train_labels = extract_features_and_labels(model, dataloader_tr)
    train_tsne = tsne.fit_transform(train_features)
    plot_data(train_tsne, train_labels)

    # 1.2 Extract test features
    test_features, test_labels = extract_features_and_labels(model, dataloader_te)

    test_tsne = tsne.fit_transform(test_features)
    plot_data(test_tsne, test_labels)





path='.\\checkpoints\\demo_autocast_debug_style15IAMcharH32rmPunct_GANres16_bs128\\5_net_Style_Encoder.pth'
#pretrain = torch.load()
opt = TrainOptions().parse()
print(opt)
opt.dataname ='style15IAMcharH32rmPunct'
opt.batch_size=128
opt.dataset_mode='style'
torch.backends.cudnn.benchmark = True
device = "cuda"
tr_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
tr_dataset_size = len(tr_dataset)
print(tr_dataset_size)
opt.dataname += "_val"
opt.dataroot = dataset_catalog.datasets[opt.dataname]
opt.scaler = GradScaler()
opt.test=True
te_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
te_dataset_size = len(te_dataset)

model=StyleEncoder(opt,already_trained=True,features_only=True,path=path).cuda()
#pr=torch.load(path)
#print(pr.keys())

#net.load_state_dict(state_dict)
#model.load_state_dict(pr)
#net.load_state_dict(state_dict)
#print(list(model.vgg.modules())
#x = image_tensor
#vgg19_fearures = torch.nn.Sequential(*(list(model.vgg.children())[:9]))
#print(vgg19_fearures.modules)
#for (i, batch) in enumerate(tr_dataset):
    #print(i)
show_features(model.vgg,tr_dataset,te_dataset)
