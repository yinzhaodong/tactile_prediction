# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import utils
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SVG-ConvLSTM/saved_models/"
train_data_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/train_image_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/slip_detection/formatted_dataset/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)

lr=0.002
beta1=0.9
batch_size=32
log_dir='logs/lp'
model_dir=''
name=''
data_root='data'
optimizer='adam'
niter=300
seed=1
epoch_size=600
image_width=32
channels=15
out_channels = 3
dataset='smmnist'
n_past=10
n_future=10
n_eval=30
rnn_size=128
prior_rnn_layers=1
posterior_rnn_layers=1
predictor_rnn_layers=2
z_dim=10  # number of latent variables
g_dim=128
beta=0.0001
data_threads=5
num_digits=2
last_frame_skip='store_true'
epochs = 100

train_percentage = 0.9
validation_percentage = 0.1

features = [lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
            image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
            posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
            last_frame_skip, epochs, train_percentage, validation_percentage]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, train=True)
        dataset_validate = FullDataSet(self.data_map, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train=False, validation=False):
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(train_data_dir + value[0])

        tactile_images = []
        for image_name in np.load(train_data_dir + value[2]):
            tactile_images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


class ModelTrainer:
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        self.optimizer = optim.Adam

        import models.lstm as lstm_models
        self.frame_predictor = lstm_models.convlstm(g_dim + z_dim, g_dim, rnn_size, predictor_rnn_layers, batch_size)
        self.posterior = lstm_models.gaussian_lstm(g_dim, z_dim, rnn_size, posterior_rnn_layers, batch_size)
        self.prior = lstm_models.gaussian_lstm(g_dim, z_dim, rnn_size, prior_rnn_layers, batch_size)
        self.frame_predictor.apply(utils.init_weights)

        import models.dcgan_32 as model
        self.encoder = model.encoder(g_dim, channels, out_channels)
        self.decoder = model.decoder(g_dim, channels, out_channels)
        self.encoder.apply(utils.init_weights)
        self.decoder.apply(utils.init_weights)

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=lr, betas=(beta1, 0.999))
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=lr, betas=(beta1, 0.999))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=lr, betas=(beta1, 0.999))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=lr, betas=(beta1, 0.999))

        self.mae_criterion = nn.L1Loss()

        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

    def train(self, tactiles, actions):
        states = actions[0, :, :]
        state_action = torch.cat((torch.cat(20*[states.unsqueeze(0)], 0), actions), 2)
        state_action_image = torch.cat(32*[torch.cat(32*[state_action.unsqueeze(3)], axis=3).unsqueeze(4)], axis=4)
        x = torch.cat((state_action_image, tactiles), 2)

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        # initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        mae = 0
        kld = 0
        for i in range(1, n_past + n_future):
            h = self.encoder(x[i - 1])
            h_target = self.encoder(x[i])[0]
            if last_frame_skip or i < n_past:
                h, skip = h
            else:
                h = h[0]
            z_t, mu, logvar = self.posterior(h_target)
            _, mu_p, logvar_p = self.prior(h)
            h_pred = self.frame_predictor(torch.cat([h, torch.cat(8*[torch.cat(8*[z_t.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)], 1))
            x_pred = self.decoder([h_pred, skip])
            mae += self.mae_criterion(x_pred, x[i][:, 12:, :, :])
            kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)

        loss = mae + kld * beta
        loss.backward()

        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return mae.data.cpu().numpy() /(n_past + n_future), kld.data.cpu().numpy() /(n_future + n_past)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) +(torch.exp(logvar1) +(mu1 - mu2) ** 2) /(2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / batch_size

    def train_full_model(self):
        self.frame_predictor.train()
        self.posterior.train()
        self.prior.train()
        self.encoder.train()
        self.decoder.train()

        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            epoch_mae_losses = 0.0
            epoch_kld_losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                if batch_features[1].shape[0] == 32:
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    mae, kld = self.train(tactiles=tactile, actions=action)
                    epoch_mae_losses += mae.item()
                    epoch_kld_losses += kld.item()
                    if index:
                        mean_kld = epoch_kld_losses / index
                        mean_mae = epoch_mae_losses / index
                    else:
                        mean_kld = 0.0
                        mean_mae = 0.0

                    progress_bar.set_description("epoch: {}, ".format(epoch) + "MAE: {:.4f}, ".format(float(mae.item())) + "kld: {:.4f}, ".format(float(kld.item())) + "mean MAE: {:.4f}, ".format(mean_mae)  + "mean kld: {:.4f}, ".format(mean_kld))
                    progress_bar.update()
                break
            plot_training_loss.append([mean_mae, mean_kld])

            # Validation checking:
            val_mae_losses = 0.0
            val_kld_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == 32:
                        tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                        action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                        val_mae, val_kld = self.train(tactiles=tactile, actions=action)
                        val_mae_losses += val_mae.item()
                        val_kld_losses += val_kld.item()
                break
            plot_validation_loss.append([val_mae_losses / index__, val_kld_losses / index__])
            print("Validation mae: {:.4f}, ".format(val_mae_losses / index__) +  " || Validation kld: {:.4f}, ".format(val_kld_losses / index__))

            # save the train/validation performance data
            np.save(model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # Early stopping:
            if previous_val_mean_loss < val_mae_losses / index__:
                early_stop_clock += 1
                previous_val_mean_loss = val_mae_losses / index__
                if early_stop_clock == 4:
                    print("Early stopping")
                    break
            else:
                if best_val_loss > val_mae_losses / index__:
                    print("saving model")
                    # save the model
                    torch.save({'encoder': self.encoder, 'decoder': self.decoder, 'frame_predictor': self.frame_predictor,
                                'posterior': self.posterior, 'prior': self.prior, 'features': features}, model_save_path + "SVG-ConvLSTM")

                    best_val_loss = val_mae_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_mae_losses / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_full_model()

