# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import numpy as np

from pickle import load
from pickle import dump
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTP/saved_models/model_22_09_2021_16_19/"
test_data_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/test_image_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/"

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

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
        tactile_data = np.load(test_data_dir + value[1])
        experiment_number = np.load(test_data_dir + value[3])
        time_steps = np.load(test_data_dir + value[4])
        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps]


class ACTP(nn.Module):
    def __init__(self):
        super(ACTP, self).__init__()
        self.lstm1 = nn.LSTM(48 + 12, 200).to(device)  # tactile
        self.lstm2 = nn.LSTM(200, 200).to(device)  # tactile
        self.fc1 = nn.Linear(200 + 48, 200).to(device)  # tactile + pos
        self.fc2 = nn.Linear(200, 48).to(device)  # tactile + pos
        self.tan_activation = nn.Tanh().to(device)

    def forward(self, tactiles, actions):
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        outputs = []
        hidden1 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
        hidden2 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))

        for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
            # 2. Run through lstm:
            if index > context_frames-1:
                out4 = out4.squeeze()
                tiled_action_and_state = torch.cat((actions.squeeze()[index+1], state), 1)
                action_and_tactile = torch.cat((out4, tiled_action_and_state), 1)
                out1, hidden1 = self.lstm1(action_and_tactile.unsqueeze(0), hidden1)
                out2, hidden2 = self.lstm2(out1, hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                outputs.append(out4.squeeze())
            else:
                tiled_action_and_state = torch.cat((actions.squeeze()[index+1], state), 1)
                action_and_tactile = torch.cat((sample_tactile, tiled_action_and_state), 1)
                out1, hidden1 = self.lstm1(action_and_tactile.unsqueeze(0), hidden1)
                out2, hidden2 = self.lstm2(out1, hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                last_output = out4

        outputs = [last_output] + outputs
        return torch.stack(outputs)


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()
        self.full_model = torch.load(model_save_path + "ACTP_model")
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
        # 2. Run through full test set and get unscaled outputs (store in array for each trial).
        # 3. Plot some trials for different taxels.
        self.test_predictions = self.find_test_predictions()
        self.losses = self.find_losses()
        self.save_trials()
        self.test_plots()

    def find_test_predictions(self):
        test_predictions = []
        for index, batch_features in enumerate(self.test_full_loader):
            action = batch_features[0].permute(1, 0, 2).to(device)
            tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)
            tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)

            experiment = batch_features[2][:, context_frames:].cpu().detach()
            time_steps = batch_features[3][:, context_frames:].cpu().detach()

            test_predictions.append([tactile_predictions.permute(1, 0, 2).cpu().detach(), tactile[context_frames:].permute(1, 0, 2).cpu().detach(), experiment, time_steps])

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
        np.save(model_save_path + "test_losses",
                np.array([["L1 loss: ", mean], ["L1 loss x: ", mean_x], ["L1 loss y: ", mean_y], ["L1 loss z: ", mean_z],
                          ["L1 loss t1: ", mean_t1], ["L1 loss t5: ", mean_t5], ["L1 loss t10: ", mean_t10]]))

        return mean, mean_x, mean_y, mean_z, mean_t1, mean_t5, mean_t10

    def save_trials(self):
        trials = dict()
        for pred, label, exp, ts in self.test_predictions:
            for batch_sequence in range(len(ts)):
                if str(exp[batch_sequence][0].item()) in trials:
                    trials[str(exp[batch_sequence])].append([pred[batch_sequence], label[batch_sequence], ts[batch_sequence]])
                else:
                    trials[str(exp[batch_sequence])] = [[pred[batch_sequence], label[batch_sequence], ts[batch_sequence]]]

        # save the outputs for each test trial:
        dump(trials, open(model_save_path + 'trials.pkl', 'wb'))

    def test_plots(self):
        trials = load(open(model_save_path + 'trials.pkl', 'rb'))
        data_to_plot = trials["0"]



if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTester()
    MT.test_full_model()
