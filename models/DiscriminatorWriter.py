from torch import nn
from torchvision.models import resnet18


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