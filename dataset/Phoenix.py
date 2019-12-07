import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Phoenix(Dataset):
    def __init__(self, path, mode, transforms=None):
        super(Phoenix, self).__init__()
        self.path = path
        self.transforms = transforms

        self.input_images = os.listdir(f'{path}/rgb')
        self.seg_images = os.listdir(f'{path}/semseg_color')

    def __getitem__(self, idx):
        img = cv2.imread(f'{self.path}/rgb/{self.input_images[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_img = cv2.imread(f'{self.path}/semseg_color/{self.seg_images[idx]}')
        seg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {
            'img': img,
            'segLabel': seg_img,
            'exist': None,
            'img_name': self.input_images[idx]
        }
        """if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            exist = np.array(self.exist_list[idx])
        else:
            segLabel = None
            exist = None"""

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return min(len(self.input_images), len(self.seg_images))

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            #exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            #exist = [b['exist'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'exist': None,
                   'img_name': [x['img_name'] for x in batch]}

        return samples
