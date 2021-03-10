import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class StyleDataset(Dataset):
    """Style Handwriting dataset."""

    def get_writer(self, directory, dr):
        path = self.xml_dir + "\\" + dr + ".xml"
        with open(path, 'r') as xml:
            writer_id = xml.readlines()[4].split('writer-id=\"')[1][:3]
        return writer_id

    def build_style_df(self):
        category_list = os.listdir(self.root_dir)
        writer_all_files = defaultdict(list)
        for directory in category_list:
            for dr in os.listdir(f'{self.root_dir}\\{directory}'):
                writer = self.get_writer(directory, dr)
                for file in os.listdir(f'{self.root_dir}\\{directory}\\{dr}'):
                    writer_all_files[writer].append((directory, dr, file))
        columns = [*[f"img{i}" for i in range(self.k)], "writer"]
        for writer in writer_all_files:
            l = writer_all_files[writer]
            random.shuffle(l)
            chunks = [l[x:x + self.k] for x in range(0, len(l), self.k)]
            if len(chunks[-1]) != self.k:
                chunks = chunks[:-1]
            for chunk in chunks:
                chunk = ['\\'.join(x[0:3]) for x in chunk]
                # print(chunk, len(chunk))
                # print([*chunk, writer])
                a = pd.DataFrame([[*chunk, writer]], columns=columns)
                # print(a)
                self.style_df = self.style_df.append(a, ignore_index=True)

    def __init__(self, root_dir, xml_path, transform=None, k=15):
        """
        Args:
            root_dir (string): Directory with all the images.
            xml_path (string): Directory with all the xmls.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.xml_dir = xml_path
        self.transform = transform
        self.k = k
        self.style_df = pd.DataFrame(columns=[*[f"img{i}" for i in range(self.k)], "writer"])
        self.build_style_df()

    def __len__(self):
        return len(self.style_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_names = [os.path.join(self.root_dir,
                                  self.style_df.iloc[idx, i]) for i in range(self.k)]
        images = [io.imread(img_names[i]) for i in range(self.k)]
        writer = self.style_df.iloc[idx, self.k]
        sample = {'images': images, 'writer': writer}

        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    s = StyleDataset('C:\\Users\\User\\Documents\\Handwiting GAN project\\IAM\\words', 'C:\\Users\\User\\Documents\\Handwiting GAN project\\IAM\\xml')
    print(s[0])