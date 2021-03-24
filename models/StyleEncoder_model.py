from torch import nn
import torch
from torchvision.models import vgg19_bn
from torchvision.models import vgg16
from torchvision.models import resnet18
# import dominate
import os


class StyleEncoder(nn.Module):

    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model

    def get_parmas_to_optimize(self):
        params_to_update = []
        for name, param in self.vgg.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        return params_to_update

    def __init__(self, already_trained=False, path=None,device="cuda", **kwargs):
        super(StyleEncoder, self).__init__()
        self.name = 'S'
        self.device=device
        self.cur_loss = torch.zeros(1)
        self.loss = nn.CrossEntropyLoss()
        self.n_labels=3#96
        if already_trained:
            self.vgg = self.prepare_vgg_extractor(path=path)
        else:
            self.vgg = self.prepare_vgg_extractor(index_freeze=0)  # torch.load('..\\models_pth\\vgg19_bn_features.pth')
        # print(self.vgg)
        #self.vgg.parameters()
        #
        self.optimizer = torch.optim.Adam(self.get_parmas_to_optimize(),lr=0.005*0.1,weight_decay=0*0.05)
        #torch.optim.Adam(model.parameters(),
                                          #lr=0.001)


    def forward(self, x, *input):
        return self.vgg(x)

    def backward(self, x, *input):
        #hot_vector=torch.zeros((self.data['label'].squeeze(0).shape[0],self.n_labels)).to(self.device)
        #for i,label in enumerate(self.data['label']):
            #hot_vector[i,label]=1
        loss = self.loss(x, self.data['label'])
        self.cur_loss = loss
        loss.backward()

    def optimize(self):
        x = self.forward(self.data['style'])
        self.backward(x)

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_input(self, data):
        self.data = data
        self.data['style']=self.data['style'].to(self.device)
        self.data['label'] = self.data['label'].to(self.device)

    def save_network(self, epoch):
        torch.save(self.vgg.state_dict(), f"checkpoints/vgg{epoch}")


    def prepare_vgg_extractor(self,index_freeze=40, path=""):
        # option to load our-trained vgg
        if path == "":
            vgg19 = model_ft = vgg19_bn(pretrained=True).to(self.device) ##.to("cpu")  # .eval()resnet18(pretrained=True)
            #print(vgg19.modules)
            freeze_network(vgg19, 0)
            replace_head(vgg19, self.n_labels)
        else:
            vgg19 = torch.load(path).to(self.device)
            freeze_network(vgg19)
        """model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
        """
        vgg19 = vgg19.to(self.device)
        # print(dir(vgg19))
        print(vgg19.modules)
        # modules=list(resnet152.children())[:-1]
        # resnet152=nn.Sequential(*modules)
        #vgg19_fearures = torch.nn.Sequential(*(list(vgg19.children())[0][:-1]))
        # print(vgg19_fearures.modules)
        # x = torch.zeros([1, 3, 224, 224]).to("cpu")
        # print(vgg19(x))
        # print(vgg19_fearures(x).shape)

        # print("features,", vgg19.features)
        # print("classfifier,", vgg19.classifier)
        # print("paarmas,", list(vgg19.parameters()))
        # print([(i, 1) for i, f in enumerate(vgg19.parameters()) if f.requires_grad])
        # print(sum([1 for f in vgg19.classifier if f.requires_grad]))
        child_counter = 0
        """for child in vgg19.children():
            print(" child", child_counter, "is:")
            print(child)
            print(f'len is {len(list(child.parameters()))}')
            child_counter += 1"""
        # see that one reqgrad and the other doesnt
        # print(list(vgg19.features[37].parameters()))
        # print("*" * 60)
        # print(list(vgg19.features[38].parameters()))
        # print("*"*60)
        # print(list(vgg19.features[40].parameters()))
        return vgg19


def freeze_network(net, index=None):
    if index is None:
        for param in net.parameters():
            param.requires_grad = False
    else:
        # TODO- unsdertand with there are more params than layers
        for layer in net.features[:index]:
            for param in layer.parameters():
                param.requires_grad = False


def replace_head(model, num_writers,name="vgg"):
    # replace the output of 1000 pictures with num_wrtiers layer
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_writers)
    #num_ftrs = model.fc.in_features
    #model.fc=nn.Linear(num_ftrs, num_writers)
    # input_size = 224


# prepare_vgg_extractor(40)
