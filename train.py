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
import numpy as np
from torchvision import transforms
import torch.utils.data
from loss import Loss
import pytorch_ssim
import lwtd_net

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

    dehaze_net = lwtd_net.LWTD().cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.train_val_loader(config.train_clean_images_path, config.train_haze_images_path)
    val_dataset = dataloader.train_val_loader(config.val_clean_images_path, config.val_haze_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = Loss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dehaze_net.train()
    max_ssim = 0
    for epoch in range(config.num_epochs):
        global_loss_train = 0
        for iteration, (img_clean, img_haze) in enumerate(train_loader):

            img_clean = img_clean.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            loss, l1_loss, edge_loss, texture_loss = criterion.forward(clean_image, img_clean)
            loss = criterion(clean_image, img_clean)

            global_loss_train += loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":",
                      loss.item(), '|l1 loss:', l1_loss, '|edge loss: ', edge_loss, '|texture loss: ', texture_loss)
                print("Loss at iteration", iteration + 1, ":", loss.item())

            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        ssim_scores = 0

        with torch.no_grad():
            for iter_val, (img_clean, img_haze) in enumerate(val_loader):
                img_clean = img_clean.cuda()
                img_haze = img_haze.cuda()

                clean_image = dehaze_net(img_haze)

                torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_clean), 0),
                                             config.sample_output_folder + str(iter_val + 1) + ".jpg")

                ssim = pytorch_ssim.ssim(clean_image, img_clean)
                ssim_scores += ssim
        torch.cuda.empty_cache()
        print(ssim_scores / (iter_val + 1))
        with open('./snapshots/log.txt', 'a') as f:
            f.write(str(epoch) + '  ' + str(global_loss_train / (len(train_loader))) + '  ' + str(ssim_scores / (
                    iter_val + 1)) + '\n')

        if ssim_scores / (iter_val + 1) > max_ssim:
            max_ssim = ssim_scores / (iter_val + 1)
            torch.save(dehaze_net.state_dict(), 'snapshots/best_v3_qua.pth')
        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")
        print('epoch ' + str(epoch + 1) + ' done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--clean_images_path', type=str,
                        default='/home/zeb/PythonProjects/Dehazing/Datasets/GZU/foggy_tianhetan_highway/images/clean/')
    parser.add_argument('--hazy_images_path', type=str, default='/home/zeb/PythonProjects/Dehazing/Datasets/GZU/foggy_tianhetan_highway/images/hazy/')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
