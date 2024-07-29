import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2


class train_val_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path):
        self.haze_list = os.listdir(hazy_images_path)
        self.data_list = []
        # print(len(self.haze_list))
        if 'Thumbs.db' in self.haze_list:

            self.haze_list.remove('Thumbs.db')
        for item in self.haze_list:
            # print(item)
            # keyname = item.split('_')[0] + '_' + item.split('_')[1] + '.jpg'
            keyname = item.split('_')[0] + '.png'
            # keyname = item.split('_foggy_')[0] + '.png'
            self.data_list.append([orig_images_path + keyname, hazy_images_path + item])
        print("Total validation examples:", len(self.haze_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path).convert('RGB')
        data_hazy = Image.open(data_hazy_path).convert('RGB')

        #
        # data_orig = data_orig.resize((480, 640), Image.ANTIALIAS)
        # data_hazy = data_hazy.resize((480, 640), Image.ANTIALIAS)

        # data_orig = data_orig.resize((720, 540), Image.ANTIALIAS)
        # data_hazy = data_hazy.resize((720, 540), Image.ANTIALIAS)

        data_orig = data_orig.resize((640, 480), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((640, 480), Image.ANTIALIAS)

        # data_orig = data_orig.resize((1280, 720), Image.ANTIALIAS)
        # data_hazy = data_hazy.resize((1280, 720), Image.ANTIALIAS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
