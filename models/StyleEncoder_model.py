import contextlib

from torch import nn
import torch
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import vgg19_bn
from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50
# import dominate
import os


class StyleEncoder(nn.Module):

    def load_state_dict(self, state_dict,
                        strict= True):
        self.vgg.load_state_dict(state_dict)
    def load_checkpoint(self,filepath):
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

    def __init__(self, opt,already_trained=False, path=None,device="cuda", **kwargs):
        super(StyleEncoder, self).__init__()
        self.name = 'S'
        self.gpu_ids = opt.gpu_ids
        self.device=device
        self.cur_loss = torch.zeros(1).to(self.device)
        self.val_loss = torch.zeros(1).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.n_labels=140#396
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if already_trained:
            self.vgg = self.prepare_vgg_extractor(path=path)
        else:
            self.vgg = self.prepare_vgg_extractor(index_freeze=0)  # torch.load('..\\models_pth\\vgg19_bn_features.pth')
        # print(self.vgg)
        #self.vgg.parameters()
        #
        self.optimizer = torch.optim.Adam(self.get_parmas_to_optimize(),lr=0.005*0.1,weight_decay=0.05*1)
        self.mixed=opt.autocast_bit
        self.scaler =opt.scaler
        #= opt.scaler =
        #torch.optim.Adam(model.parameters(),
                                          #lr=0.001)


    def forward(self, x, *input, save_loss=False, train=True):
        output = self.vgg(x)
        if save_loss:
            loss = self.loss(output, self.data['label']).to(self.device)
            if train:
                self.cur_loss += loss
            else:
                self.val_loss += loss
        return output

    def backward(self, *input):
        if self.mixed:
            scale = self.scaler.scale
            factor_scale= lambda x: [item * 1./self.scaler.get_scale() for item in x]
        else:
            scale = lambda x: x
        #hot_vector=torch.zeros((self.data['label'].squeeze(0).shape[0],self.n_labels)).to(self.device)
        #for i,label in enumerate(self.data['label']):
            #hot_vector[i,label]=1
        with autocast() if self.mixed else contextlib.nullcontext():
            x = self.forward(self.data['style'])
            loss = self.loss(x, self.data['label']).to(self.device)
            self.cur_loss += loss
        scale(loss).backward()

    def optimize(self):
        self.backward()

    def optimize_step(self):
        if self.mixed:
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def set_input(self, data):
        self.data = data
        self.data['style']=self.data['style'].to(self.device)
        self.data['label'] = self.data['label'].to(self.device)

    def save_network(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        name = "Style_Encoder"
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(self.save_dir, save_filename)
        net = self.vgg

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            # torch.save(net.module.cpu().state_dict(), save_path)
            if len(self.gpu_ids) > 1:
                torch.save(net.module.cpu().state_dict(), save_path)
            else:
                torch.save(net.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def zero_loss(self):
        self.cur_loss = torch.zeros(1).to(self.device)
        self.val_loss = torch.zeros(1).to(self.device)
    def prepare_vgg_extractor(self,index_freeze=40, path="",name="resnet"):
        # option to load our-trained vgg
        if path == "":
            if name=="vgg":
                vgg19 =vgg19_bn(pretrained=True)
            elif name=="resnet":
                vgg19 = resnet18(pretrained=True)
            vgg19=vgg19.to(self.device)# ##.to("cpu")  # .eval()
            #print(vgg19.modules)
            freeze_network(vgg19, 4,name=name)
            replace_head(vgg19, self.n_labels,name=name)
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


def freeze_network(net, index=None,name="vgg"):
    if index is None:
        for param in net.parameters():
            param.requires_grad = False
    else:
        print("FREEZING NETWORK=============================")
        if name=="vgg":
            # TODO- unsdertand with there are more params than layers
            for layer in net.features[:index]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif name=="resnet":
            #               0, 1   2 ,  3       4       5       6       7       8   9
            #strcutre is conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
            for layer in list(net.children())[:7]:
                print(layer)
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            raise Exception("no model name")


def replace_head(model, num_writers,name="vgg"):
    # replace the output of 1000 pictures with num_wrtiers layer
    if name=="vgg":
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_writers)
    elif name=="resnet":
        num_ftrs = model.fc.in_features
        model.fc=nn.Linear(num_ftrs, num_writers)
    else:
        raise Exception("no model name")
    # input_size = 224


# prepare_vgg_extractor(40)
