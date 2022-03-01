# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import numpy as np
import math

from pickle import load
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)


class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size ()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)



class tester:
    def __init__(self, model_path, model_path_latent, data_save_path, test_data_dir, scaler_dir):
        self.number_of_trials = [i for i in range(52)]
        # self.object_list = [9, 7, 11, 11, 7, 11, 7, 9, 11, 11, 7, 7, 7, 7, 11, 9, 11, 11, 9, 11, 9, 9]
        self.object_list = [12, 10, 4, 13, 17, 7, 16, 7, 7, 10, 14, 17, 13, 4, 16, 7, 16, 4, 14, 10, 7, 5, 10, 13, 14,
                            5, 4, 13, 4, 4, 17, 12, 7, 10, 7, 7, 5, 5, 16, 16, 12, 13, 7, 7, 14, 17, 13, 12, 17, 17, 13, 14]

        self.seen = []
        for i in self.object_list:
            if i == 11 or i == 9:
                self.seen.append('0')
            else:
                self.seen.append('1')

        self.performance_data = []
        self.image_height = 64
        self.image_width = 64

        self.model_path = model_path
        self.scaler_dir = scaler_dir
        self.test_data_dir = test_data_dir
        self.data_save_path = data_save_path
        self.model_path_latent = model_path_latent

        self.criterion = nn.L1Loss()
        self.load_scalars()

        for trial, object, seen_item in zip(self.number_of_trials, self.object_list, self.seen):
            gt = np.load(data_save_path + 'train_gt_trial_' + str(trial) + '.npy')
            pd = np.load(data_save_path + 'train_pd_trial_' + str(trial) + '.npy')
            self.calc_trial_loss(pd, gt, object, seen_item)
            # pd_descaled, gt_descaled = self.scale_back(pd, gt)
        self.save_losses()

    def calc_trial_loss(self, prediction_data, groundtruth_data, object, seen):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=64)
        psnr_calc = PSNR()

        # create images here:
        prediction_data = torch.tensor(prediction_data)
        groundtruth_data = torch.tensor(groundtruth_data)

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set_pred, batch_set_gt in zip(prediction_data, groundtruth_data):
                index += 1

                ## MAE:
                mae_loss += self.criterion(batch_set_pred, batch_set_gt).item()
                mae_loss_1 += self.criterion(batch_set_pred[0], batch_set_gt[0]).item()
                mae_loss_5 += self.criterion(batch_set_pred[4], batch_set_gt[4]).item()
                mae_loss_10 += self.criterion(batch_set_pred[9], batch_set_gt[9]).item()
                # images_gt = []
                # images_pd = []
                # for ts in range(10):
                #     images_gt.append(self.create_image(batch_set_gt[ts]))
                #     images_pd.append(self.create_image(batch_set_pred[ts]))
                # images_gt = torch.tensor(np.array(images_gt))
                # images_pd = torch.tensor(np.array(images_pd))
                #
                # ## SSIM:
                # ssim_loss += ssim_calc(images_pd, images_gt)
                # ssim_loss_1 += ssim_calc(images_pd[0].unsqueeze(0), images_gt[0].unsqueeze(0))
                # ssim_loss_5 += ssim_calc(images_pd[4].unsqueeze(0), images_gt[4].unsqueeze(0))
                # ssim_loss_10 += ssim_calc(images_pd[9].unsqueeze(0), images_gt[9].unsqueeze(0))
                # ## PSNR:
                # psnr_loss += psnr_calc(images_pd, images_gt)
                # psnr_loss_1 += psnr_calc(images_pd[0], images_gt[0])
                # psnr_loss_5 += psnr_calc(images_pd[4], images_gt[4])
                # psnr_loss_10 += psnr_calc(images_pd[9], images_gt[9])

        self.performance_data.append(
            [mae_loss / index, mae_loss_1 / index, mae_loss_5 / index, mae_loss_10 / index, seen, object])
             # float(ssim_loss / index), float(ssim_loss_1 / index), float(ssim_loss_5 / index), float(ssim_loss_10 / index),
             # float(psnr_loss / index), float(psnr_loss_1 / index), float(psnr_loss_5 / index), float(psnr_loss_10 / index), seen, object])

        print(self.performance_data)

    def save_losses(self):
        self.performance_data = np.array(self.performance_data).astype(np.float32)
        performance_data_full = []
        performance_data_full.append(["train loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])

        # performance_data_full.append(["train loss MAE(L1) Seen objects: ", (sum([i[0] for i in self.performance_data if i[-2] == 1.0]) / len([i[0] for i in self.performance_data if i[-2] == 1.0]))])
        # performance_data_full.append(["train loss MAE(L1) Novel objects: ", (sum([i[0] for i in self.performance_data if i[-2] == 0.0]) / len([i[0] for i in self.performance_data if i[-2] == 0.0]))])

        self.objects = [i for i in range(30)]

        self.objects = list(set(self.objects))
        for object in self.objects:
            try:
                performance_data_full.append(["train loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == int(object)]) / len([i[0] for i in self.performance_data if i[-1] == int(object)]))])
            except:
                print("object {} doesnt exist".format(str(object)))

        # performance_data_full.append(["train loss SSIM: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss SSIM pred ts 1: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss SSIM pred ts 5: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss SSIM pred ts 10: ", (sum([i[7] for i in self.performance_data]) / len(self.performance_data))])

        # performance_data_full.append(["train loss SSIM Seen objects: ", (sum([i[4] for i in self.performance_data if i[-2] == 1.0]) / len([i[4] for i in self.performance_data if i[-2] == 1.0]))])
        # performance_data_full.append(["train loss SSIM Novel objects: ", (sum([i[4] for i in self.performance_data if i[-2] == 0.0]) / len([i[4] for i in self.performance_data if i[-2] == 0.0]))])

        # performance_data_full.append(["train loss PSNR: ", (sum([i[8] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss PSNR pred ts 1: ", (sum([i[9] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss PSNR pred ts 5: ", (sum([i[10] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["train loss PSNR pred ts 10: ", (sum([i[11] for i in self.performance_data]) / len(self.performance_data))])

        # performance_data_full.append(["train loss PSNR Seen objects: ", (sum([i[8] for i in self.performance_data if i[-2] == 1.0]) / len([i[8] for i in self.performance_data if i[-2] == 1.0]))])
        # performance_data_full.append(["train loss PSNR Novel objects: ", (sum([i[8] for i in self.performance_data if i[-2] == 0.0]) / len([i[8] for i in self.performance_data if i[-2] == 0.0]))])

        for i in performance_data_full:
            print(i)

        np.save(self.data_save_path + "train_losses.npy", np.array(performance_data_full))

    def scale_back(self, tactile_data, groundtruth_data):
        pt_descalled_data = []
        gt_descalled_data = []

        for time_step in range(tactile_data.shape[0]):
            (ptx, pty, ptz) = np.split(tactile_data[time_step], 3, axis=1)
            (gtx, gty, gtz) = np.split(groundtruth_data[time_step], 3, axis=1)

            xela_ptx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(ptx)
            xela_pty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(pty)
            xela_ptz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(ptz)
            xela_ptx_inverse_full = self.scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
            xela_pty_inverse_full = self.scaler_ty.inverse_transform(xela_pty_inverse_minmax)
            xela_ptz_inverse_full = self.scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
            pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))

            xela_gtx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(gtx)
            xela_gty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(gty)
            xela_gtz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(gtz)
            xela_gtx_inverse_full = self.scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
            xela_gty_inverse_full = self.scaler_ty.inverse_transform(xela_gty_inverse_minmax)
            xela_gtz_inverse_full = self.scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
            gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))

        return np.array(pt_descalled_data), np.array(gt_descalled_data)

    def load_scalars(self):
        self.scaler_tx = load(open(self.scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load(open(self.scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load(open(self.scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

    def create_image(self, tactile_data):
        # convert tactile data into an image:
        image = np.zeros((4, 4, 3), np.float32)
        index = 0
        tactile_x = tactile_data[0:16]
        tactile_y = tactile_data[16:32]
        tactile_z = tactile_data[32:48]
        for x in range(4):
            for y in range(4):
                image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
                index += 1
        reshaped_image = cv2.resize(image.astype(np.float32), dsize=(self.image_height, self.image_width), interpolation=cv2.INTER_CUBIC)
        return reshaped_image


def main():
    model_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/20220223-111825-CDNA-8/"
    model_path_latent = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SV2P/saved_models/box_only_model_14_12_2021_12_24/SV2P_model_latent_model"
    data_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/20220223-111825-CDNA-8/"
    test_data_dir = ""
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"
    tt = tester(model_path, model_path_latent, data_save_path, test_data_dir, scaler_dir)


if __name__ == "__main__":
    main()
