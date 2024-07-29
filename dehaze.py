import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import sys
import argparse
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import lwtd_net


if __name__ == '__main__':
    image_path = '/home/zeb/PythonProjects/Dehazing/Datasets/URHI/'
    dehaze_net = lwtd_net.LWTD().cuda()
    weight_path = './snapshots/best_v3_qua.pth'
    save_dir = '/home/zeb/PythonProjects/Dehazing/Results/URHI/'
    dehaze_net.load_state_dict(torch.load(weight_path))

    if os.path.isdir(image_path):
        filenames = os.listdir(image_path)
        with torch.no_grad():
            for filename in filenames:
                filepath = image_path + filename
                data_hazy = Image.open(filepath)
                data_hazy = (np.asarray(data_hazy) / 255.0)
                data_hazy = torch.from_numpy(data_hazy).float()
                data_hazy = data_hazy.permute(2, 0, 1)
                data_hazy = data_hazy.cuda().unsqueeze(0)
                clean_image = dehaze_net(data_hazy)
                torchvision.utils.save_image(clean_image, save_dir + filename)
                print(str(filenames.index(filename) + 1) + '/' + str(len(filenames)) + ' done!')
                torch.cuda.empty_cache()
