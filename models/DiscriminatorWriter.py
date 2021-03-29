from torch import nn
from torchvision.models import resnet18



# based on Encoder code from discimantor
from models.BigGAN_networks import Discriminator


class DiscriminatorWriter(Discriminator):
    def __init__(self, opt, output_dim, **kwargs):
        super(DiscriminatorWriter, self).__init__(**vars(opt))
        self.output_layer = nn.Sequential(self.activation,
                                          nn.Conv2d(self.arch['out_channels'][-1], output_dim, kernel_size=(4,2), padding=0, stride=2))

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        out = self.output_layer(h)
        return out
    #def
class DiscriminatorWriter(nn.Module):
    def __init__(self,num_writers):
        self.model=resnet18(pretrained=False)
        self.num_writers=num_writers
        #replace_head()
        #freeze_network
        pass
    def forward(self,x):
        return self.model(x)
    def parameters(self, recurse: bool = True):
        return self.model.parameters()