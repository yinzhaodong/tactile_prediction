# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import numpy as np
from tqdm import tqdm

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

model_save_path = "/home/willmandil/Robotics/tactile_prediction/tactile_prediction/models/SV2P/saved_models/box_only_"
train_data_dir = "/home/willmandil/Robotics/Data_sets/box_only_dataset/train_image64_dataset_10c_10h/"
scaler_dir = "/home/willmandil/Robotics/Data_sets/box_only_dataset/scalar_info/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)

lr = 0.001
seed = 42
mfreg = 0.1
seqlen = 20
krireg = 0.1
klreg_max = 0.1
n_masks = 10
indices = (0.9, 0.1)
max_epoch = 100
batch_size = 64
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

hp = np.array([lr, seed, seqlen, mfreg, krireg, klreg_max, n_masks, max_epoch,
               batch_size, in_channels, state_action_channels, context_frames,
               latent_channels, train_percentage, validation_percentage,
               num_iterations_1st_stage, num_iterations_2nd_stage, num_iterations_3rd_stage, linear_increase])
np.save(model_save_path + "HyperParameters", hp)

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
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, train=True)
        dataset_validate = FullDataSet(self.data_map, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=12)
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


class CDNATrainer():
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        self.net = cdna.CDNA(in_channels, state_action_channels+latent_channels, n_masks, with_generator=False).to(device)
        self.latent_model = LatentModel(batch_size, latent_channels, seqlen).to(device)
        # self.learned_prior_model = PosteriorInferenceNet(tbatch=seqlen).to(device)

        self.stat_names = 'predloss', 'kernloss', 'maskloss', 'loss'
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = {'L1': nn.L1Loss(), 'L2': nn.MSELoss(),
                          'DSSIM': DSSIM(self.net.in_channels),
                         }[criterion_name].to(device)
        self.kernel_criterion = RotationInvarianceLoss().to(device)

    def run(self):
        plot_training_loss = []
        plot_validation_loss = []
        for index_stage, self.training_stage in enumerate(["generative_network", "infurence_network", "divergence_reduction"]):
            num_iterations = 0
            self.klreg = 0
            end_of_stage = False
            progress_bar = tqdm(range(0, max_epoch), total=(max_epoch*len(self.train_full_loader)))
            for epoch in progress_bar:
                total_losses = 0.0
                total_losses_val = 0.0
                best_val_loss = 100.0

                # training:
                self.net.train()
                self.latent_model.train()

                for index, batch_features in enumerate(self.train_full_loader):
                    num_iterations += 1
                    if index_stage == 0 and num_iterations > num_iterations_1st_stage:
                        print("END OF TRAINING STAGE 1")
                        end_of_stage = True
                        break
                    if index_stage == 1 and num_iterations > num_iterations_2nd_stage:
                        print("END OF TRAINING STAGE 2")
                        end_of_stage = True
                        break
                    if index_stage == 2:
                        self.klreg = linear_step * num_iterations
                        if self.klreg > klreg_max:
                            self.klreg = klreg_max

                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    if batch_features[0].shape[0] == batch_size:
                        loss, kernloss, maskloss, total_loss = self.training_pass(tactile, action)
                        total_losses += total_loss
                        sequences = index

                        progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.5f}, ".format(float(total_loss)) + "mean loss: {:.5f}, ".format(total_losses / (index + 1)))
                        progress_bar.update()

                plot_training_loss.append(total_losses / sequences)

                for index, batch_features in enumerate(self.valid_full_loader):
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    if batch_features[0].shape[0] == batch_size:
                        loss, kernloss, maskloss, total_loss = self.training_pass(tactile, action, validation=True)
                        total_losses_val += total_loss
                        sequences = index

                total_losses_val_mean = total_losses_val / sequences
                plot_validation_loss.append(total_losses_val_mean)

                plot_validation_loss.append(total_losses_val_mean)
                print("Validation mean loss: {:.4f}, ".format(total_losses_val_mean))

                # save the train/validation performance data
                np.save(model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
                np.save(model_save_path + "plot_training_loss", np.array(plot_training_loss))

                if best_val_loss > total_losses_val_mean:
                    print("saving model")
                    torch.save(self.net, model_save_path + "SV2P_model_net")
                    torch.save(self.latent_model, model_save_path + "SV2P_model_latent_model")
                    best_val_loss = total_losses_val_mean

                if end_of_stage == True:
                    break

    def training_pass(self, tactiles, actions, validation=False):
        hidden = None
        outputs = []
        state = actions[0].to(device)

        # Now start to use the prior network:
        samples = torch.normal(0.0, 1.0, size=(batch_size, latent_channels, 8, 8)).to(device)
        if self.training_stage == "generative_network":
            prior_sample = samples
            pMean, pSTD = None, None
        elif self.training_stage == "infurence_network" or "divergence_reduction":
            if validation:
                # change to sampling pMean and pSTD from the learned_prior_model
                prior_sample = samples
                pMean, pSTD = None, None
            else:
                pMean, pSTD = self.latent_model(frames=tactiles)
                prior_sample = pMean + torch.exp(pSTD / 2.0) * samples

        if validation:
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
        else:
            for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
                state_action = torch.cat((state, sample_action), 1)
                tsa = torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3)
                tsap = torch.cat([prior_sample, tsa], axis=1)
                if index > context_frames - 1:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net(predictions_t, conditions=tsap, hidden_states=hidden)
                    outputs.append(predictions_t)
                else:
                    predictions_t, hidden, cdna_kerns_t, masks_t = self.net(sample_tactile, conditions=tsap, hidden_states=hidden)
                    last_output = predictions_t

        outputs = [last_output] + outputs

        loss = 0.0
        kernloss = 0.0
        maskloss = 0.0
        KLD_loss = 0.0
        for prediction_t, target_t in zip(outputs, tactiles[context_frames:]):
            loss_t, kernloss_t, maskloss_t, KLD_loss_t = self.__compute_loss(prediction_t, cdna_kerns_t, masks_t, target_t, pSTD, pMean)
            loss += loss_t
            kernloss += kernloss_t
            maskloss += maskloss_t
            KLD_loss += KLD_loss_t
        total_loss = (loss + kernloss + maskloss + KLD_loss) / context_frames

        if not validation:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return(loss.detach().cpu().item() / seqlen,
               kernloss.detach().cpu().item() / seqlen,
               maskloss.detach().cpu().item() / seqlen,
               total_loss.detach().cpu().item())

    def __compute_loss(self, predictions_t, cdna_kerns_t, masks_t, targets_t, pSTD, pMean):
        loss_t = self.criterion(predictions_t, targets_t)
        prior_loss = self.klreg * self.kl_divergence(pMean, pSTD)
        kernloss_t = krireg * self.kernel_criterion(cdna_kerns_t)
        maskloss_t = mfreg * masks_t[:, 1:].reshape(-1, masks_t.size(-2) * masks_t.size(-1)).abs().sum(1).mean()
        return loss_t, kernloss_t, maskloss_t, prior_loss

    def kl_divergence(self, mu, log_sigma):
        """KL divergence of diagonal gaussian N(mu,exp(log_sigma)) and N(0,1).
        """
        if mu == None:
            return 0.0
        else:
            return torch.mean(-.5 * torch.sum(1. + log_sigma - torch.square(mu) - torch.exp(log_sigma), axis=1))

if __name__ == '__main__':
    BG = BatchGenerator()
    Trainer = CDNATrainer()
    Trainer.run()

