import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2


if __name__ == '__main__':

    video_dir = '/home/zeb/PythonProjects/Dehazing/Datasets/GZU/foggy_tianhetan_highway/videos/'
    filenames = os.listdir(video_dir)
    save_dir = '/home/zeb/PythonProjects/Dehazing/Datasets/Results/GZU/foggy_tianhetan_highway/videos/'
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

    for filename in filenames:
        video_path = video_dir + filename
        save_path = save_dir + filename
        VideoCapture = cv2.VideoCapture(video_path)

        # VideoCapture.open(video_path)
        fps = VideoCapture.get(cv2.CAP_PROP_FPS)
        frames = VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        VideoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        ct = time.time()

        for i in range(int(frames)):
            # while VideoCapture.isOpened():
            ret, frame = VideoCapture.read()
            data_hazy = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            data_hazy = data_hazy / 255.0
            data_hazy = torch.from_numpy(data_hazy).float()
            data_hazy = data_hazy.permute(2, 0, 1)
            data_hazy = data_hazy.cuda().unsqueeze(0)
            ft = time.time()
            clean_frame = dehaze_net(data_hazy)
            frame_costs = time.time() - ft
            print('video ' + str(filenames.index(filename) + 1) + '/' + str(len(filenames)) + ', frame ' + str(
                i + 1) + '/' + str(int(frames)) + ' costs ' + str(frame_costs) + ' s')
            torchvision.utils.save_image(clean_frame, './frame.png')
            dehazed = cv2.imread('./frame.png')
            VideoWriter.write(dehazed)
        cost = time.time() - ct

        VideoCapture.release()
        VideoWriter.release()

        print('video ' + filename + ' done! ' + str(cost) + 's costs, ' + str(filenames.index(filename)) + '/' + str(
            len(filenames)))

