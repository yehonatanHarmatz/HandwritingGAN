import torch
from torchvision.models import vgg19_bn
from torchvision.models import vgg16
import dominate
import os
#pretrain = torch.load('..\\models_pth\\vgg19_bn_features.pth')
#print(pretrain.keys())
#print(dir(pretrain))
#vgg16 = vgg16(pretrained=True)
def vgg_extractor(path=""):
    # option to load our-trained vgg
    if path=="":
        vgg19 = vgg19_bn(pretrained=True).to("cuda").eval()
    else:
        vgg19 = torch.load(path)
    """model.half()  # convert to half precision
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    """
    freeze_network(vgg19)
    #print(dir(vgg19))
    print(vgg19.modules)
    # modules=list(resnet152.children())[:-1]
    # resnet152=nn.Sequential(*modules)
    vgg19_fearures = torch.nn.Sequential(*(list(vgg19.children())[0][:-1]))
    print(vgg19_fearures.modules)
    x = torch.zeros([1, 3, 400, 600]).to("cuda")
    # print(vgg19(x))
    print(vgg19_fearures(x).shape)
    return vgg19
def freeze_network(net):
  '''
  The function freezes all layers of a pytorch network (net).
  '''
  ######################
  ### YOUR CODE HERE ###
  ######################
  for param in net.parameters():
    param.requires_grad = False
#vgg16 = vgg16(pretrained=True).to("cuda").eval()
#freeze_network(vgg19)
#print(dir(vgg19))
#print(vgg16.modules)
vgg_extractor()

#torch.save(pretrain,'..\\models_pth\\vgg19_bn_features.pth')