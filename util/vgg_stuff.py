
#pretrain = torch.load('..\\models_pth\\vgg19_bn_features.pth')
#print(pretrain.keys())
#print(dir(pretrain))
#vgg16 = vgg16(pretrained=True)

#vgg16 = vgg16(pretrained=True).to("cuda").eval()
#freeze_network(vgg19)
#print(dir(vgg19))
#print(vgg16.modules)
vgg_extractor()

#torch.save(pretrain,'..\\models_pth\\vgg19_bn_features.pth')