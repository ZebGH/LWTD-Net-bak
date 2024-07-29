import torch
import torch.nn as nn
import numpy as np
import pytorch_ssim
from torch.autograd import Variable
import cv2


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def get_ssim(self, dehazed, orig):
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        return 1 - ssim_loss(orig, dehazed)


# class QELoss(nn.Module):
#     def __init__(self):
#         super(QELoss, self).__init__()
#
#     def img_var(self, img, mean):
#         m, n = np.shape(img)[-2:]
#         var = torch.sqrt(torch.sum(img - mean) ** 2) / (m * n + 1)
#         return var
#
#     def img_cov(self, dehazed, mean_dehazed, orig, mean_orig):
#         m, n = np.shape(orig)[-2:]
#         cov = torch.sum((orig - mean_orig) * (dehazed - mean_dehazed)) / (m * n - 1)
#         return cov
#
#     def img_MSE(self, dehazed, orig):
#         m, n = np.shape(orig)[-2:]
#         MSE = torch.sum(torch.sqrt((dehazed - orig) ** 2)) / (m * n)
#         return MSE
#
#     def forward(self, dehazed, orig):
#         MES = torch.mean((orig - dehazed) ** 2)
#         PSNR = 10 * torch.log10(255 ** 2 / MES)
#         # PSNR = torch.from_numpy(PSNR)
#
#         c1 = (0.01 * 255) ** 2
#         c2 = (0.03 * 255) ** 2
#         mean_ori = torch.mean(orig)
#         mean_dehazed = torch.mean(dehazed)
#         var_ori = self.img_var(orig, mean_ori)
#         var_dehazed = self.img_var(dehazed, mean_dehazed)
#         cov = self.img_cov(dehazed, mean_dehazed, orig, mean_ori)
#         SSIM = (2 * mean_ori + 2 * mean_dehazed + c1) * (2 * cov + c2) / (
#                 (mean_dehazed ** 2 + mean_ori ** 2 + c1) * (var_ori ** 2 + var_dehazed ** 2 + c2))
#
#         return 0.5 * 53.4100 / PSNR + 0.5 * 1 / SSIM
#
#         # return 53.4100 / PSNR

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def get_l1_loss(self, dehazed, orig):
        m, n = np.shape(orig)[-2:]
        l1_loss = torch.sum(torch.abs(orig - dehazed)) / (m * n * 3)
        return l1_loss / orig.shape[0]

    # def make_orders
    def tensors2numpys(self, tensors):
        tensors = tensors.permute(2, 3, 1, 0)
        tensors = torch.asarray(tensors * 255, dtype=torch.uint8)
        tensors = tensors.cpu()
        numpy_tensor= tensors.detach().numpy()

        return numpy_tensor

    def get_edge_loss(self, pred, orig):
        # print(np.shape(orig))
        # canny = cv2.Canny(orig, 100, 125, 5)
        canny = cv2.Canny(orig, 90, 100, 5)
        canny = cv2.dilate(canny, (3, 3), iterations=1)
        # shape = canny.shape

        mask = np.zeros((480, 640, 3), dtype=np.uint8)

        mask[:, :, 0] = canny
        mask[:, :, 1] = canny
        mask[:, :, 2] = canny
        mask_pred = cv2.bitwise_and(pred, mask)
        mask_orig = cv2.bitwise_and(orig, mask)
        mask_orig = torch.asarray(mask_orig)
        mask_pred = torch.asarray(mask_pred)
        ground = cv2.countNonZero(canny) * 3
        # ground = orig.shape[0] * orig.shape[1] * 3
        edge_loss = torch.sum(torch.abs(mask_orig - mask_pred)) / (ground * 255)
        return edge_loss

    def get_texture_loss(self, pred, orig):
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        lap = cv2.dilate(lap, (3, 3), iterations=1)
        lap3 = np.zeros((480, 640, 3), dtype=np.uint8)
        lap3[:, :, 0] = lap
        lap3[:, :, 1] = lap
        lap3[:, :, 2] = lap
        mask_pred = cv2.bitwise_and(pred, lap3)
        mask_orig = cv2.bitwise_and(orig, lap3)
        mask_orig = torch.asarray(mask_orig)
        mask_pred = torch.asarray(mask_pred)
        ground = cv2.countNonZero(lap) * 3

        # orig = torch.asarray(orig)
        # pred = torch.asarray(pred)
        # texture_loss = torch.sum(lap3 * torch.abs(mask_orig - mask_pred)) / (ground * 255)

        texture_loss = torch.sum(torch.abs(mask_orig - mask_pred)) / (ground * 255)
        return texture_loss

    def forward(self, pred, orig):
        pred_imgs = self.tensors2numpys(pred)
        orig_imgs = self.tensors2numpys(orig)
        # print(type(orig))
        loss = 0
        v_loss = 0
        for i in range(orig.shape[0]):
            loss += self.get_edge_loss(pred_imgs[:, :, :, i], orig_imgs[:, :, :, i])
            v_loss += self.get_texture_loss(pred_imgs[:, :, :, i], orig_imgs[:, :, :, i])
        self.edge_loss = loss / orig.shape[0]
        self.l1_loss = self.get_l1_loss(orig, pred)
        self.texture_loss = v_loss / orig.shape[0]

        return self.l1_loss + self.edge_loss + self.texture_loss, float(self.l1_loss), float(self.edge_loss), float(self.texture_loss)
