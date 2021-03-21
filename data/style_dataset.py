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
        parser.add_argument('--aug_dataroot', type=str, default=None,
                            help='augmentation images file location, default is None (no augmentation)')
        parser.add_argument('--aug', action='store_true', default=False,
                            help='use augmentation (currently relevant for OCR training)')
        parser.add_argument('--k', type=int, default=15,
                            help='number of images in the style object')
        return parser

    def __init__(self, opt, target_transform=None):

        BaseDataset.__init__(self, opt)

        self.k = opt.k

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

        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.target_transform = target_transform
        # if opt.collate:
        #     self.collate_fn = TextCollator(opt)
        # else:
        #     self.collate_fn = RegularCollator(opt)

        self.labeled = opt.labeled

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index,device='cuda'):
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
            for imgbuf in style:
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = ToTensor()(Image.open(buf)).to(device)
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]
                if self.transform is not None and False: #TODO
                    img = self.transform(img)
                imgs.append(img)
            # print([image for image in imgs])
            # imgs_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(image) for image in imgs], batch_first=True)
            # imgs_tensor = concat_images([torch.flatten(image, 0, 1) for image in imgs])
            imgs_tensor = concat_images(imgs)
            item = {'style': imgs_tensor, 'imgs_path': style_key, 'idx':index}
            # im = tensor2im(imgs_tensor.unsqueeze(0))
            # img = Image.fromarray(im, 'RGB')
            # img.save('my.png')
            # img.show()
            if self.labeled:
                label_key = 'label-%09d' % index
                label = int(txn.get(label_key.encode('utf-8')).decode())
                # label = int(style['label'])
                if self.target_transform is not None:
                    label = self.target_transform(label)
                item['label'] = label
                words_key = 'words-%09d' % index
                words = np.load(io.BytesIO(txn.get(words_key.encode('utf-8'))))
                item['words'] = str(words)

            if hasattr(self,'Z'):
                z = self.Z[index-1]
                item['z'] = z

        return item