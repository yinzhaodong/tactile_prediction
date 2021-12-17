# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import math
import numpy as np
from tqdm import tqdm

from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import sv2p.cdna as cdna
from sv2p.ssim import DSSIM
from sv2p.ssim import Gaussian2d
from sv2p.model import PosteriorInferenceNet, LatentModel
from sv2p.criteria import RotationInvarianceLoss

model_path      = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SV2P/saved_models/box_only_model_14_12_2021_12_24/"
data_save_path  = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SV2P/saved_models/box_only_model_14_12_2021_12_24/"
test_data_dir   = "/home/user/Robotics/Data_sets/box_only_dataset/test_image_dataset_10c_10h_64/"
scaler_dir      = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info/"

train = False

lr = 0.001
seed = 42
mfreg = 0.1
seqlen = 20
krireg = 0.1
klreg_max = 0.1
n_masks = 10
indices = (0.9, 0.1)
max_epoch = 100
batch_size = 32
in_channels = 3
state_action_channels = 12
context_frames = 10
latent_channels = 10
train_percentage = 0.9
# latent_std_min = 0.005
validation_percentage = 0.1

linear_increase = 20000 / 10
num_iterations_1st_stage = 100000 / 10
num_iterations_2nd_stage = 50000 / 10
num_iterations_3rd_stage = 50000 / 10

(lr, seed, seqlen, mfreg, krireg, klreg_max, n_masks, max_epoch,
batch_size, in_channels, state_action_channels, context_frames,
latent_channels, train_percentage, validation_percentage,
num_iterations_1st_stage, num_iterations_2nd_stage, num_iterations_3rd_stage, linear_increase) = np.load(model_path + "HyperParameters.npy")

batch_size = int(batch_size)
seed = int(seed)
context_frames = int(context_frames)
state_action_channels = int(state_action_channels)
in_channels = int(in_channels)
n_masks = int(n_masks)
latent_channels = int(latent_channels)
seqlen = int(seqlen)

linear_step = klreg_max / linear_increase

device = "cuda"
warm_start = False
dataset_name = "BoxOnly"
criterion_name = "L1"
scheduled_sampling_k = False

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map):
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(test_data_dir + value[0])

        tactile_images = []
        for image_name in np.load(test_data_dir + value[2]):
            tactile_images.append(np.load(test_data_dir + image_name))

        experiment_number = np.load(test_data_dir + value[3])
        time_steps = np.load(test_data_dir + value[4])
        meta = test_data_dir + value[5]
        return [robot_data.astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps, meta]


class ModelTester():
    def __init__(self):
        self.test_full_loader = BG.load_full_data()

        # load model:
        self.net = torch.load(model_path + "SV2P_model_net.zip").to(device)
        self.net.eval()

        self.latent_model = torch.load(model_path + "SV2P_model_latent_model.zip").to(device)
        self.latent_model.eval()

        self.load_scalars()

        self.stat_names = 'predloss', 'kernloss', 'maskloss', 'loss'
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = {'L1': nn.L1Loss(), 'L2': nn.MSELoss(),
                          'DSSIM': DSSIM(self.net.in_channels),
                          }[criterion_name].to(device)
        self.kernel_criterion = RotationInvarianceLoss().to(device)

    def test_full_model(self):
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.tg = []
        self.tp1 = []
        self.tp5 = []
        self.tp10 = []
        self.current_exp = 0
        self.objects = []

        for index, batch_features in enumerate(self.test_full_loader):
            tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            tactile_predictions = self.SV2P_pass_through(tactile, action)

            experiment_number = batch_features[2].permute(1, 0)[context_frames:]
            time_steps = batch_features[3].permute(1, 0)[context_frames:]
            self.meta = batch_features[4][0]

            current_batch = 0
            new_batch = 0
            for index, exp in enumerate(experiment_number.T):
                if exp[0] == self.current_exp:
                    current_batch += 1
                else:
                    new_batch += 1

            for i in [0, 1]:
                if i == 0:
                    tactile_cut = tactile[:, 0:current_batch, :, :, :]
                    tactile_predictions_cut = tactile_predictions[:, 0:current_batch, :, :, :]
                    experiment_number_cut = experiment_number[:, 0:current_batch]
                    time_steps_cut = time_steps[:, 0:current_batch]
                if i == 1:
                    tactile_cut = tactile[:, current_batch:, :, :, :]
                    tactile_predictions_cut = tactile_predictions[:, current_batch:, :, :, :]
                    experiment_number_cut = experiment_number[:, current_batch:]
                    time_steps_cut = time_steps[:, current_batch:]

                self.prediction_data.append(
                    [tactile_predictions_cut.cpu().detach(), tactile_cut[context_frames:].cpu().detach(),
                     experiment_number_cut.cpu().detach(), time_steps_cut.cpu().detach()])

                # convert back to 48 feature tactile readings for plotting:
                gt = []
                p1 = []
                p5 = []
                p10 = []
                for batch_value in range(tactile_predictions_cut.shape[1]):
                    gt.append(cv2.resize(
                        tactile_cut[context_frames - 1][batch_value].permute(1, 2, 0).cpu().detach().numpy(),
                        dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    p1.append(cv2.resize(
                        tactile_predictions_cut[0][batch_value].permute(1, 2, 0).cpu().detach().numpy(),
                        dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    p5.append(cv2.resize(
                        tactile_predictions_cut[4][batch_value].permute(1, 2, 0).cpu().detach().numpy(),
                        dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    p10.append(cv2.resize(
                        tactile_predictions_cut[9][batch_value].permute(1, 2, 0).cpu().detach().numpy(),
                        dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())

                gt = np.array(gt)
                p1 = np.array(p1)
                p5 = np.array(p5)
                p10 = np.array(p10)
                descalled_data = []

                # print(gt.shape, p1.shape, p5.shape, p10.shape, gt.shape[0])
                if gt.shape[0] != 0:
                    for data in [gt, p1, p5, p10]:
                        (tx, ty, tz) = np.split(data, 3, axis=1)
                        xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(tx)
                        xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(ty)
                        xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(tz)
                        xela_x_inverse_full = self.scaler_tx.inverse_transform(xela_x_inverse_minmax)
                        xela_y_inverse_full = self.scaler_ty.inverse_transform(xela_y_inverse_minmax)
                        xela_z_inverse_full = self.scaler_tz.inverse_transform(xela_z_inverse_minmax)
                        descalled_data.append(np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1))

                    self.tg_back_scaled.append(descalled_data[0])
                    self.tp1_back_scaled.append(descalled_data[1])
                    self.tp5_back_scaled.append(descalled_data[2])
                    self.tp10_back_scaled.append(descalled_data[3])

                    self.tg.append(gt)
                    self.tp1.append(p1)
                    self.tp5.append(p5)
                    self.tp10.append(p10)

                if i == 0 and new_batch != 0:
                    print("currently testing trial number: ", str(self.current_exp))
                    # self.calc_train_trial_performance()
                    self.calc_trial_performance()

                    self.save_predictions(self.current_exp, scaled=True)
                    self.tg_back_scaled = self.tg
                    self.tp1_back_scaled = self.tp1
                    self.tp5_back_scaled = self.tp5
                    self.tp10_back_scaled = self.tp10
                    self.save_predictions(self.current_exp, scaled=False)

                    self.prediction_data = []
                    self.tg_back_scaled = []
                    self.tp1_back_scaled = []
                    self.tp5_back_scaled = []
                    self.tp10_back_scaled = []
                    self.tg = []
                    self.tp1 = []
                    self.tp5 = []
                    self.tp10 = []

                    self.current_exp += 1
                if i == 0 and new_batch == 0:
                    break

        self.calc_test_performance()
        # self.calc_train_performance()

    def SV2P_pass_through(self, tactiles, actions):
        hidden = None
        outputs = []
        state = actions[0].to(device)

        samples = torch.normal(0.0, 1.0, size=(batch_size, latent_channels, 8, 8)).to(device)
        prior_sample = samples
        pMean, pSTD = None, None

        with torch.no_grad():
            for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                tsap = torch.cat([prior_sample, tsa], axis=1)
                if index > context_frames-1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net(predictions_t, conditions=tsap, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net(sample_tactile, conditions=tsap, hidden_states=hidden)
                    last_output = predictions_t

        outputs = [last_output] + outputs

        return torch.stack(outputs)


    def calc_train_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(
            ["test loss MAE(L1): ",(sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ",(sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ",(sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ",(sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ",(sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ",(sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ",(sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])

        [print(i) for i in performance_data_full]
        np.save(data_save_path + 'TRAIN_model_performance_loss_data', np.asarray(performance_data_full))

    def save_predictions(self, experiment_to_test, scaled, train):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t1 = []
        trial_predicted_data_t5 = []
        trial_predicted_data_t10 = []
        for index in range(len(self.tg_back_scaled)):
            for batch_number in range(len(self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t1.append(self.tp1_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append(self.tp5_back_scaled[index][batch_number])
                    trial_predicted_data_t10.append(self.tp10_back_scaled[index][batch_number])
                    trial_groundtruth_data.append(self.tg_back_scaled[index][batch_number])

        if scaled:
            add = "SCALED_"
        else:
            add = "DESCALED_"

        if train:
            tadd = "train_"
        else:
            tadd = "test_"

        plot_save_dir = data_save_path + add + "test_plots_" + str(experiment_to_test)
        try:
            os.mkdir(plot_save_dir)
        except:
            "directory already exists"

        np.save(plot_save_dir + '/trial_groundtruth_data', np.array(trial_groundtruth_data))
        np.save(plot_save_dir + '/prediction_data_t1', np.array(trial_predicted_data_t1))
        np.save(plot_save_dir + '/prediction_data_t5', np.array(trial_predicted_data_t5))
        np.save(plot_save_dir + '/prediction_data_t10', np.array(trial_predicted_data_t10))

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        np.save(plot_save_dir + '/meta_data', np.array(meta_data))

    def calc_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=64)
        psnr_calc = PSNR()

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion(batch_set[0], batch_set[1]).item()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss += mae_loss_check
                    mae_loss_1 += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5 += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x += self.criterion(batch_set[0][:, :, 0], batch_set[1][:, :, 0]).item()
                    mae_loss_y += self.criterion(batch_set[0][:, :, 1], batch_set[1][:, :, 1]).item()
                    mae_loss_z += self.criterion(batch_set[0][:, :, 2], batch_set[1][:, :, 2]).item()
                    ## SSIM:
                    for i in range(len(batch_set)):
                        index_ssim += 1
                        ssim_loss += ssim_calc(batch_set[0][i], batch_set[1][i])
                    ssim_loss_1 += ssim_calc(batch_set[0][0], batch_set[1][0])
                    ssim_loss_5 += ssim_calc(batch_set[0][4], batch_set[1][4])
                    ssim_loss_10 += ssim_calc(batch_set[0][9], batch_set[1][9])
                    ssim_loss_x += ssim_calc(batch_set[0][:, :, 0], batch_set[1][:, :, 0])
                    ssim_loss_y += ssim_calc(batch_set[0][:, :, 1], batch_set[1][:, :, 1])
                    ssim_loss_z += ssim_calc(batch_set[0][:, :, 2], batch_set[1][:, :, 2])
                    ## PSNR:
                    psnr_loss += psnr_calc(batch_set[0], batch_set[1])
                    psnr_loss_1 += psnr_calc(batch_set[0][0], batch_set[1][0])
                    psnr_loss_5 += psnr_calc(batch_set[0][4], batch_set[1][4])
                    psnr_loss_10 += psnr_calc(batch_set[0][9], batch_set[1][9])
                    psnr_loss_x += psnr_calc(batch_set[0][:, :, 0], batch_set[1][:, :, 0])
                    psnr_loss_y += psnr_calc(batch_set[0][:, :, 1], batch_set[1][:, :, 1])
                    psnr_loss_z += psnr_calc(batch_set[0][:, :, 2], batch_set[1][:, :, 2])

        self.performance_data.append(
            [mae_loss / index, mae_loss_1 / index, mae_loss_5 / index, mae_loss_10 / index, mae_loss_x / index,
             mae_loss_y / index, mae_loss_z / index, ssim_loss / index_ssim, ssim_loss_1 / index,
             ssim_loss_5 / index, ssim_loss_10 / index, ssim_loss_x / index, ssim_loss_y / index,
             ssim_loss_z / index, psnr_loss / index, psnr_loss_1 / index, psnr_loss_5 / index,
             psnr_loss_10 / index, psnr_loss_x / index, psnr_loss_y / index, psnr_loss_z / index, seen, object])
        self.objects.append(object)

        print(mae_loss / index, object, meta_data_file_name)

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append (
            ["test loss MAE(L1): ", (sum ([i[0] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 1: ",
                                       (sum ([i[1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 5: ",
                                       (sum ([i[2] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 10: ",
                                       (sum ([i[3] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force x: ",
                                       (sum ([i[4] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force y: ",
                                       (sum ([i[5] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force z: ",
                                       (sum ([i[6] for i in self.performance_data]) / len (self.performance_data))])

        for trial, values in enumerate (self.performance_data):
            performance_data_full.append (["test loss MAE(L1) Trial " + str (trial) + "; object: " + str (
                self.performance_data[trial][-1]) + ": ", self.performance_data[trial][0]])

        self.objects = list (set (self.objects))
        for object in self.objects:
            performance_data_full.append (["test loss MAE(L1) Trained object " + str (object) + ": ", (
                        sum ([i[0] for i in self.performance_data if i[-1] == str (object)]) / len (
                    [i[0] for i in self.performance_data if i[-1] == str (object)]))])

        performance_data_full.append (["test loss MAE(L1) Seen objects: ", (
                    sum ([i[0] for i in self.performance_data if i[-2] == '1']) / len (
                [i[0] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss MAE(L1) Novel objects: ", (
                    sum ([i[0] for i in self.performance_data if i[-2] == '0']) / len (
                [i[0] for i in self.performance_data if i[-2] == '0']))])

        performance_data_full.append (
            ["test loss SSIM: ", (sum ([i[7] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred ts 1: ",
                                       (sum ([i[8] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred ts 5: ",
                                       (sum ([i[9] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred ts 10: ", (
                    sum ([i[10] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred force x: ", (
                    sum ([i[11] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred force y: ", (
                    sum ([i[12] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss SSIM pred force z: ", (
                    sum ([i[13] for i in self.performance_data]) / len (self.performance_data))])

        performance_data_full.append (["test loss SSIM Seen objects: ", (
                    sum ([i[7] for i in self.performance_data if i[-2] == '1']) / len (
                [i[7] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss SSIM Novel objects: ", (
                    sum ([i[7] for i in self.performance_data if i[-2] == '0']) / len (
                [i[7] for i in self.performance_data if i[-2] == '0']))])

        performance_data_full.append (
            ["test loss PSNR: ", (sum ([i[14] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred ts 1: ", (
                    sum ([i[15] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred ts 5: ", (
                    sum ([i[16] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred ts 10: ", (
                    sum ([i[17] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred force x: ", (
                    sum ([i[18] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred force y: ", (
                    sum ([i[19] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss PSNR pred force z: ", (
                    sum ([i[20] for i in self.performance_data]) / len (self.performance_data))])

        performance_data_full.append (["test loss PSNR Seen objects: ", (
                    sum ([i[14] for i in self.performance_data if i[-2] == '1']) / len (
                [i[14] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss PSNR Novel objects: ", (
                    sum ([i[14] for i in self.performance_data if i[-2] == '0']) / len (
                [i[14] for i in self.performance_data if i[-2] == '0']))])

        [print (i) for i in performance_data_full]
        np.save (data_save_path + 'model_performance_loss_data', np.asarray (performance_data_full))

    def load_scalars(self):
        self.scaler_tx = load (open (scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load (open (scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load (open (scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load (open (scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load (open (scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load (open (scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean ((img1 - img2) ** 2)
        return 20 * torch.log10 (255.0 / torch.sqrt (mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor (
        [exp (-(x - window_size // 2) ** 2 / float (2 * sigma ** 2)) for x in range (window_size)])
    return gauss / gauss.sum ()

def create_window(window_size, channel):
    _1D_window = gaussian (window_size, 1.5).unsqueeze (1)
    _2D_window = _1D_window.mm (_1D_window.t ()).float ().unsqueeze (0).unsqueeze (0)
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

if __name__ == "__main__":
    BG = BatchGenerator ()
    MT = ModelTester ()
    MT.test_full_model ()
