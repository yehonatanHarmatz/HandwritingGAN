import io
import os
import six
import sys
import ast
import lmdb
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from data import BaseDataset
from data.base_dataset import get_transform
from PIL import Image

from util.util import binary_to_dict, concat_images, tensor2im


class StyleDataset(BaseDataset):
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--collate', action='store_false', default=True,
        #                     help='use regular collate function in data loader')


        return parser

    def __init__(self, opt, target_transform=None):

        BaseDataset.__init__(self, opt)

        self.k = opt.k
        self.min_load = opt.min_load
        self.env = lmdb.open(
            os.path.abspath(opt.dataroot),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (opt.dataroot))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
            self.nSamples = nSamples
            self.mapping_id = {i:k for i,k in enumerate(ast.literal_eval(txn.get('writers_mapping_id'.encode('utf-8')).decode('utf-8')))}

        if opt.aug and opt.aug_dataroot is not None:
            self.env_aug = lmdb.open(
                os.path.abspath(opt.aug_dataroot),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

            with self.env_aug.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
                self.nSamples = self.nSamples + nSamples
                self.nAugSamples = nSamples

        self.transform = get_transform(opt)
        self.transform_org = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.target_transform = target_transform
        # if opt.collate:
        #     self.collate_fn = TextCollator(opt)
        # else:
        #     self.collate_fn = RegularCollator(opt)

        self.labeled = opt.labeled
        self.device = opt.device

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        envAug = False
        if hasattr(self, 'env_aug'):
            if index>=self.nAugSamples:
                index = index-self.nAugSamples
            else:
                envAug = True
        index += 1
        with eval('self.env'+'_aug'*envAug+'.begin(write=False)') as txn:
            style_key = 'style-%09d' % index
            a = txn.get(style_key.encode('utf-8'))
            style = np.load(io.BytesIO(a))
            imgs = []
            org_size = []
            for imgbuf in style[:self.k]:
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    im = Image.open(buf)
                    if not self.min_load:
                        im2 = im
                    im = im.resize((im.size[0], 224//self.k))
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]
                img = ToTensor()(im).to(self.device)
                if not self.min_load:
                    img2 = ToTensor()(im2).to(self.device)
                    org_size.append(img2)
                    imgs_tensor_org = concat_images(org_size, normalized=('Normalize' in str(self.transform)),
                                                    result_h=sum(org.shape[1] for org in org_size))

                imgs.append(img)
            # print([image for image in imgs])
            # imgs_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(image) for image in imgs], batch_first=True)
            # imgs_tensor = concat_images([torch.flatten(image, 0, 1) for image in imgs])
            imgs_tensor = concat_images(imgs, normalized=('Normalize' in str(self.transform)))
            if self.transform is not None:
                img_pil=torchvision.transforms.ToPILImage()(imgs_tensor)
                imgs_tensor = self.transform(img_pil).to(self.device)
                if not self.min_load:
                    img_pil_org=torchvision.transforms.ToPILImage()(imgs_tensor_org).convert('L')
                    imgs_tensor_org = self.transform_org(img_pil_org).to(self.device)
                #print(self.transform)
            # im = tensor2im(imgs_tensor.unsqueeze(0))
            # img = Image.fromarray(im, 'RGB')
            # img.resize((img.size[0], 224))
            # img.show()
            # imgs_tensor = ToTensor()(im).to(self.device)
            # imgs_tensor = torchvision.transforms.Normalize([0,0,0], [1,1,1], inplace=False)(imgs_tensor)
            if not self.min_load:
                item = {'style': imgs_tensor, 'imgs_path': style_key, 'idx':index, 'original':imgs_tensor_org}
            else:
                item = {'style': imgs_tensor}
            # im = tensor2im(imgs_tensor.unsqueeze(0))
            # img = Image.fromarray(im, 'RGB')
            # img.save('my.png')
            # img.show()
            if self.labeled:
                label_key = 'writer-%09d' % index
                label = int(txn.get(label_key.encode('utf-8')).decode())
                # label = int(style['label'])
                if self.target_transform is not None:
                    label = self.target_transform(label)
                item['label'] = label
                if not self.min_load:
                    words_key = 'words-%09d' % index
                    words = np.load(io.BytesIO(txn.get(words_key.encode('utf-8'))))
                    item['words'] = str(list(words))

            if hasattr(self,'Z'):
                z = self.Z[index-1]
                item['z'] = z

        return item