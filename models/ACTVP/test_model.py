# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import numpy as np

from pickle import load
from pickle import dump
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTVP/saved_models/model_23_09_2021_15_31/"
test_data_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/test_image_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/"

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

train_percentage = 0.9
validation_percentage = 0.1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


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
        return [robot_data.astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return(torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device))


class ACTVP(nn.Module):
    def __init__(self):
        super(ACTVP, self).__init__()
        self.convlstm1 = ConvLSTMCell(input_dim=3, hidden_dim=12, kernel_size=(3, 3), bias=True).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=24, hidden_dim=24, kernel_size=(3, 3), bias=True).cuda()
        self.conv1 = nn.Conv2d(in_channels=27, out_channels=3, kernel_size=5, stride=1, padding=2).cuda()

    def forward(self, tactiles, actions):
        self.batch_size = actions.shape[1]
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        outputs = []
        for index,(sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
            sample_tactile.to(device)
            sample_action.to(device)
            # 2. Run through lstm:
            if index > context_frames-1:
                hidden_1, cell_1 = self.convlstm1(input_tensor=output, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, output), 1)
                output = self.conv1(skip_connection_added)
                outputs.append(output)
            else:
                hidden_1, cell_1 = self.convlstm1(input_tensor=sample_tactile, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, sample_tactile), 1)
                output = self.conv1(skip_connection_added)
                last_output = output
        outputs = [last_output] + outputs
        return torch.stack(outputs)


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()
        self.full_model = torch.load(model_save_path + "ACTVP_model")
        self.criterion = nn.L1Loss()

    def load_scalers(self):
        tactile_standard_scaler_x = load(open(scaler_dir + 'tactile_standard_scaler_x.pkl', 'rb'))
        tactile_standard_scaler_y = load(open(scaler_dir + 'tactile_standard_scaler_y.pkl', 'rb'))
        tactile_standard_scaler_z = load(open(scaler_dir + 'tactile_standard_scaler_z.pkl', 'rb'))
        tactile_min_max_scalar_x = load(open(scaler_dir + 'tactile_min_max_scalar_x.pkl', 'rb'))
        tactile_min_max_scalar_y = load(open(scaler_dir + 'tactile_min_max_scalar_y.pkl', 'rb'))
        tactile_min_max_scalar_z = load(open(scaler_dir + 'tactile_min_max_scalar.pkl', 'rb'))

        robot_min_max_scalar_px = load(open(scaler_dir + 'robot_min_max_scalar_px.pkl', 'rb'))
        robot_min_max_scalar_py = load(open(scaler_dir + 'robot_min_max_scalar_py.pkl', 'rb'))
        robot_min_max_scalar_pz = load(open(scaler_dir + 'robot_min_max_scalar_pz.pkl', 'rb'))
        robot_min_max_scalar_ex = load(open(scaler_dir + 'robot_min_max_scalar_ex.pkl', 'rb'))
        robot_min_max_scalar_ey = load(open(scaler_dir + 'robot_min_max_scalar_ey.pkl', 'rb'))
        robot_min_max_scalar_ez = load(open(scaler_dir + 'robot_min_max_scalar_ez.pkl', 'rb'))

    def test_full_model(self):
        # To Do:
        # 1. Run through full test set and get loss values + loss values for specific aspects.
        # 2. Run through full test set and get unscaled outputs(store in array for each trial).
        # 3. Plot some trials for different taxels.
        self.test_predictions = self.find_test_predictions()
        self.losses = self.find_losses()
        self.save_trials()
        self.test_plots()

    def find_test_predictions(self):
        test_predictions = []
        for index, batch_features in enumerate(self.test_full_loader):
            tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)

            # descale back down to 48 features:
            taxel_predictions = self.create_image_to_taxels(tactile_predictions.permute(0, 1, 4, 3, 2))
            taxel_groundtruth = self.create_image_to_taxels(tactile[context_frames:].permute(0, 1, 4, 3, 2))

            experiment = batch_features[2][:, context_frames:].cpu().detach()
            time_steps = batch_features[3][:, context_frames:].cpu().detach()

            test_predictions.append([taxel_predictions, taxel_groundtruth, experiment, time_steps])

        return test_predictions

    def find_losses(self):
        losses, losses_x, losses_y, losses_z, losses_t1, losses_t5, losses_t10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for pred, label, exp, ts in self.test_predictions:
            losses += self.criterion(pred, label).item()
            losses_x += self.criterion(pred, label[:, :, 0:16]).item()
            losses_y += self.criterion(pred, label[:, :, 16:32]).item()
            losses_z += self.criterion(pred, label[:, :, 32:48]).item()
            losses_t1 += self.criterion(pred, label[:, 0, :]).item()
            losses_t5 += self.criterion(pred, label[:, 4, :]).item()
            losses_t10 += self.criterion(pred, label[:, 9, :]).item()

        mean = losses / len(self.test_predictions)
        mean_x = losses_x / len(self.test_predictions)
        mean_y = losses_y / len(self.test_predictions)
        mean_z = losses_z / len(self.test_predictions)
        mean_t1 = losses_t1 / len(self.test_predictions)
        mean_t5 = losses_t5 / len(self.test_predictions)
        mean_t10 = losses_t10 / len(self.test_predictions)
        np.save(model_save_path + "test_losses", np.array([["L1 loss: ", mean], ["L1 loss x: ", mean_x], ["L1 loss y: ", mean_y], ["L1 loss z: ", mean_z],
                                                                                ["L1 loss t1: ", mean_t1], ["L1 loss t5: ", mean_t5], ["L1 loss t10: ", mean_t10]]))

        return mean, mean_x, mean_y, mean_z, mean_t1, mean_t5, mean_t10

    def save_trials(self):
        trials = dict()
        for pred, label, exp, ts in self.test_predictions:
            for batch_sequence in range(len(ts)):
                if str(exp[batch_sequence][0].item()) in trials:
                    trials[str(exp[batch_sequence])].append(
                        [pred[batch_sequence], label[batch_sequence], ts[batch_sequence]])
                else:
                    trials[str(exp[batch_sequence])] = [
                        [pred[batch_sequence], label[batch_sequence], ts[batch_sequence]]]

        # save the outputs for each test trial:
        dump(trials, open(model_save_path + 'trials.pkl', 'wb'))

    def test_plots(self):
        trials = load(open(model_save_path + 'trials.pkl', 'rb'))
        data_to_plot = trials["0"]

    def create_image_to_taxels(self, tactile_predictions):
        reshaped_images = []
        for sequence in range(tactile_predictions.shape[0]):
            reshaped_image_batch = []
            for batch in range(tactile_predictions.shape[1]):
                reshaped_image_batch.append(np.rot90(cv2.resize(np.array(tactile_predictions[sequence][batch].cpu().detach()), dsize=(4, 4), interpolation=cv2.INTER_CUBIC), k=3, axes=(0, 1)).flatten())
            reshaped_images.append(reshaped_image_batch)

        return np.array(reshaped_images)



if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTester()
    MT.test_full_model()
