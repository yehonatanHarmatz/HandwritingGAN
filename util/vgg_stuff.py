
#pretrain = torch.load('..\\models_pth\\vgg19_bn_features.pth')
#print(pretrain.keys())
#print(dir(pretrain))
#vgg16 = vgg16(pretrained=True)
import torch
#vgg16 = vgg16(pretrained=True).to("cuda").eval()
#freeze_network(vgg19)d
#print(dir(vgg19))
from torchvision.models import vgg19_bn

vgg19=vgg19_bn(pretrained=True)
vgg19.eval()
print(vgg19.modules)
x= torch.zeros((1,3,224,224))
print(vgg19(x).shape)
h=x
#print(vgg19.children()[0])
# at least 50 layers, max 52
cut= torch.nn.Sequential(*list(vgg19.children())[0][:50])
print(cut.modules)
#1,512,14,14
print(cut(x).shape)

#torch.save(pretrain,'..\\models_pth\\vgg19_bn_features.pth')