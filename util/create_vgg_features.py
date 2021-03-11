import torch
import os
print(os.popen("dir").read())
pretrain = torch.load('..\\models_pth\\vgg19_bn.pth')
print(pretrain.keys())
print(dir(pretrain))
lis=[]
# remove the last 3 layers (weight and bias) of the classifier
for key in list(pretrain.keys())[-2:]:
    pretrain.pop(key)
print(pretrain.keys())
torch.save(pretrain,'..\\models_pth\\vgg19_bn_features.pth')