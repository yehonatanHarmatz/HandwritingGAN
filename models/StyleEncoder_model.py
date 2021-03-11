from torch import nn
import torch

from util.vgg_stuff import vgg_extractor


class StyleEncoder(nn.Module):

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def __init__(self, D_ch=64, D_wide=True, **kwargs):
        super(StyleEncoder, self).__init__()
        self.name = 'S'
        self.vgg = vgg_extractor()  # torch.load('..\\models_pth\\vgg19_bn_features.pth')
        print(self.vgg)

    def forward(self, x, *input):
        return self.vgg(x)
