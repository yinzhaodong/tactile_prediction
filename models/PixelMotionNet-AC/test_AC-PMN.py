# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


model_path      = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/PixelMotionNet-AC/saved_models/prelim_model_26_10_2021_09_31/ACPixelMotionNet_model"
data_save_path  = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/PixelMotionNet-AC/saved_models/prelim_model_26_10_2021_09_31/"
test_data_dir   = "/home/user/Robotics/Data_sets/data_collection_preliminary/test_image_dataset_10c_10h/"
scaler_dir      = "/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/"

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


class ACPixelMotionNet(nn.Module):
    def __init__(self):
        super(ACPixelMotionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()
        self.convlstm1 = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=44, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
        self.upconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.upconv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.outconv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1).cuda()

        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

    def forward(self, tactiles, actions):
        self.batch_size = actions.shape[1]
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(8,8))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(8,8))
        outputs = []

        for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
            # sample_tactile.to(device)
            # sample_action.to(device)

            if index > context_frames-1:
                out1 = self.maxpool1(self.relu1(self.conv1(output)))
                out2 = self.maxpool2(self.relu2(self.conv2(out1)))

                hidden_1, cell_1 = self.convlstm1(input_tensor=out2, cur_state=[hidden_1, cell_1])

                # Add in tiled action and state:
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)

                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1(self.relu3(self.upconv1(hidden_2)))
                skip_connection = torch.cat((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2(self.relu4(self.upconv2(skip_connection)))

                output = self.tanh(self.outconv(out4))
                outputs.append(output)

            else:
                out1 = self.maxpool1(self.relu1(self.conv1(sample_tactile)))
                out2 = self.maxpool2(self.relu2(self.conv2(out1)))

                hidden_1, cell_1 = self.convlstm1(input_tensor=out2, cur_state=[hidden_1, cell_1])

                # Add in tiled action and state:
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)

                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1(self.relu3(self.upconv1(hidden_2)))
                skip_connection = torch.cat((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2(self.relu4(self.upconv2(skip_connection)))

                output = self.tanh(self.outconv(out4))
                last_output = output

        outputs = [last_output] + outputs
        return torch.stack(outputs)


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()

        # load model:
        self.full_model = ACPixelMotionNet()
        self.full_model = torch.load(model_path)
        self.full_model.eval()

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

    def test_full_model(self):
        self.prediction_data = []
        plot_training_loss = []
        for index, batch_features in enumerate(self.test_full_loader):
            tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.

            experiment_number = batch_features[2].permute(1,0)[context_frames:]
            time_steps = batch_features[3].permute(1,0)[context_frames:]
            self.prediction_data.append([tactile_predictions.cpu().detach(), tactile[context_frames:].cpu().detach(), experiment_number.cpu().detach(), time_steps.cpu().detach()])

            # convert to 48 bits
            seq_tp = []
            seq_tg = []
            seq_tg10 = []
            image_pred_t10 = tactile_predictions[-1]
            image_gt = tactile[context_frames-1]
            image_gt_10 = tactile[-1]

            tp_images.append(image_pred_t10)
            tg_images.append(image_gt)
            tg10_images.append(image_gt_10)
            for batch_value in range(len(image_pred_t10)):
                seq_tp.append(cv2.resize(image_pred_t10[batch_value].permute(1,2,0).cpu().detach().numpy(), dsize=(4,4), interpolation=cv2.INTER_CUBIC).flatten())
                seq_tg.append(cv2.resize(image_gt[batch_value].permute(1,2,0).cpu().detach().numpy(), dsize=(4,4), interpolation=cv2.INTER_CUBIC).flatten())
                seq_tg10.append(cv2.resize(image_gt_10[batch_value].permute(1,2,0).cpu().detach().numpy(), dsize=(4,4), interpolation=cv2.INTER_CUBIC).flatten())

            image_pred_t10_batch = np.array(seq_tp)
            image_gt_batch = np.array(seq_tg)
            image_gt10_batch = np.array(seq_tg10)
            (tpx, tpy, tpz) = np.split(image_pred_t10_batch, 3, axis=1)
            xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(tpx)
            xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(tpy)
            xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(tpz)
            xela_x_inverse_full = self.scaler_tx.inverse_transform(xela_x_inverse_minmax)
            xela_y_inverse_full = self.scaler_ty.inverse_transform(xela_y_inverse_minmax)
            xela_z_inverse_full = self.scaler_tz.inverse_transform(xela_z_inverse_minmax)
            tp_back_scaled.append(np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1))

            (tpx, tpy, tpz) = np.split(image_gt_batch, 3, axis=1)
            xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(tpx)
            xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(tpy)
            xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(tpz)
            xela_x_inverse_full = self.scaler_tx.inverse_transform(xela_x_inverse_minmax)
            xela_y_inverse_full = self.scaler_ty.inverse_transform(xela_y_inverse_minmax)
            xela_z_inverse_full = self.scaler_tz.inverse_transform(xela_z_inverse_minmax)
            tg_back_scaled.append(np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1))

            (tpx, tpy, tpz) = np.split(image_gt10_batch, 3, axis=1)
            xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(tpx)
            xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(tpy)
            xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(tpz)
            xela_x_inverse_full = self.scaler_tx.inverse_transform(xela_x_inverse_minmax)
            xela_y_inverse_full = self.scaler_ty.inverse_transform(xela_y_inverse_minmax)
            xela_z_inverse_full = self.scaler_tz.inverse_transform(xela_z_inverse_minmax)
            tg10_back_scaled.append(np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1))

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=32)
        psnr_calc = PSNR()

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                ## MAE:
                mae_loss    += self.criterion(batch_set[0], batch_set[1]).item()
                mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()

                ## SSIM:
                for i in range(len(batch_set)):
                    index_ssim += 1
                    ssim_loss += ssim_calc(batch_set[0][i], batch_set[1][i])
                ssim_loss_1  += ssim_calc(batch_set[0][0], batch_set[1][0])
                ssim_loss_5  += ssim_calc(batch_set[0][4], batch_set[1][4])
                ssim_loss_10 += ssim_calc(batch_set[0][9], batch_set[1][9])
                ssim_loss_x  += ssim_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                ssim_loss_y  += ssim_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                ssim_loss_z  += ssim_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])

                ## PSNR:
                psnr_loss    += psnr_calc(batch_set[0], batch_set[1])
                psnr_loss_1  += psnr_calc(batch_set[0][0], batch_set[1][0])
                psnr_loss_5  += psnr_calc(batch_set[0][4], batch_set[1][4])
                psnr_loss_10 += psnr_calc(batch_set[0][9], batch_set[1][9])
                psnr_loss_x  += psnr_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                psnr_loss_y  += psnr_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                psnr_loss_z  += psnr_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])


        performance_data = []
        performance_data.append(["test loss MAE(L1): ", (mae_loss / index)])
        performance_data.append(["test loss MAE(L1) pred ts 1: ", (mae_loss_1 / index)])
        performance_data.append(["test loss MAE(L1) pred ts 5: ", (mae_loss_5 / index)])
        performance_data.append(["test loss MAE(L1) pred ts 10: ", (mae_loss_10 / index)])
        performance_data.append(["test loss MAE(L1) pred force x: ", (mae_loss_x / index)])
        performance_data.append(["test loss MAE(L1) pred force y: ", (mae_loss_y / index)])
        performance_data.append(["test loss MAE(L1) pred force z: ", (mae_loss_z / index)])

        performance_data.append(["test loss SSIM: ", (ssim_loss / index_ssim)])
        performance_data.append(["test loss SSIM pred ts 1: ", (ssim_loss_1 / index)])
        performance_data.append(["test loss SSIM pred ts 5: ", (ssim_loss_5 / index)])
        performance_data.append(["test loss SSIM pred ts 10: ", (ssim_loss_10 / index)])
        performance_data.append(["test loss SSIM pred force x: ", (ssim_loss_x / index)])
        performance_data.append(["test loss SSIM pred force y: ", (ssim_loss_y / index)])
        performance_data.append(["test loss SSIM pred force z: ", (ssim_loss_z / index)])

        performance_data.append(["test loss PSNR: ", (psnr_loss / index)])
        performance_data.append(["test loss PSNR pred ts 1: ", (psnr_loss_1 / index)])
        performance_data.append(["test loss PSNR pred ts 5: ", (psnr_loss_5 / index)])
        performance_data.append(["test loss PSNR pred ts 10: ", (psnr_loss_10 / index)])
        performance_data.append(["test loss PSNR pred force x: ", (psnr_loss_x / index)])
        performance_data.append(["test loss PSNR pred force y: ", (psnr_loss_y / index)])
        performance_data.append(["test loss PSNR pred force z: ", (psnr_loss_z / index)])

        [print (i) for i in performance_data]
        np.save(data_save_path + 'model_performance_loss_data', np.asarray(performance_data))

    def create_test_plots(self, exp_to_test):
        # calculate tactile values for full sample:
        time_step_to_test_t1 = 0
        time_step_to_test_t5 = 4
        time_step_to_test_t9 = 5
        predicted_data_t1 = []
        predicted_data_t5 = []
        predicted_data_t9 = []
        groundtruth_data = []

        experiment_to_test = exp_to_test
        for index, batch_set in enumerate(tactile_predictions):
            for batch in range(0, len(batch_set[0])):
                if experiment == experiment_to_test:
                    prediction_values = batch_set[time_step_to_test_t1][batch]
                    predicted_data_t1.append(prediction_values)
                    prediction_values = batch_set[time_step_to_test_t5][batch]
                    predicted_data_t5.append(prediction_values)
                    prediction_values = batch_set[time_step_to_test_t9][batch]
                    predicted_data_t9.append(prediction_values)
                    gt_values = tactile_groundtruth[index][time_step_to_test_t1][batch]
                    groundtruth_data.append(gt_values)

        try:
            plot_save_dir = data_save_path + "test_plots_" + str(experiment_to_test)
            os.mkdir(plot_save_dir)
        except:
            "directory already exists"

        index = 0
        titles = ["sheerx", "sheery", "normal"]
        for j in range(3):
            for i in range(16):
                groundtruth_taxle = []
                predicted_taxel = []
                predicted_taxel_t1 = []
                predicted_taxel_t5 = []
                predicted_taxel_t10 = []
                for k in range(len(predicted_data_t1)):
                    predicted_taxel_t1.append(predicted_data_t1[k][j+i].cpu().detach().numpy())
                    predicted_taxel_t5.append(predicted_data_t5[k][j+i].cpu().detach().numpy())
                    predicted_taxel_t10.append(predicted_data_t9[k][j+i].cpu().detach().numpy())
                    groundtruth_taxle.append(groundtruth_data[k][j+i].cpu().detach().numpy())

                index += 1

                fig, ax1 = plt.subplots()
                ax1.set_xlabel('time step')
                ax1.set_ylabel('tactile reading')
                ax1.plot([None for i in range(0)] + [i for i in predicted_taxel_t1], alpha=0.5, c="b", label="t1")
                ax1.plot([None for i in range(4)] + [i for i in predicted_taxel_t5], alpha=0.5, c="k", label="t5")
                ax1.plot([None for i in range(9)] + [i for i in predicted_taxel_t10], alpha=0.5, c="g", label="t10")
                ax1.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('loss')  # we already handled the x-label with ax1
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fig.subplots_adjust(top=0.90)
                ax1.legend(loc="upper right")
                plt.title("Simple_LSTM tactile " + str(index))
                plt.savefig(plot_save_dir + '/full_test_plot_' + str(index) + '.png', dpi=300)
                plt.show()

                fig, ax1 = plt.subplots()
                ax1.set_xlabel('time step')
                ax1.set_ylabel('tactile reading')
                ax1.plot([None for i in range(0)] + [i for i in predicted_taxel_t1], alpha=0.5, c="b", label="t1")
                ax1.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('loss')  # we already handled the x-label with ax1
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fig.subplots_adjust(top=0.90)
                ax1.legend(loc="upper right")
                plt.title("Simple_LSTM tactile " + str(index))
                plt.savefig(plot_save_dir + '/T0_test_plot_' + str(index) + '.png', dpi=300)
                plt.show()

                fig, ax1 = plt.subplots()
                ax1.set_xlabel('time step')
                ax1.set_ylabel('tactile reading')
                ax1.plot([None for i in range(4)] + [i for i in predicted_taxel_t5], alpha=0.5, c="b", label="t5")
                ax1.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('loss')  # we already handled the x-label with ax1
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fig.subplots_adjust(top=0.90)
                ax1.legend(loc="upper right")
                plt.title("Simple_LSTM tactile " + str(index))
                plt.savefig(plot_save_dir + '/T5_test_plot_' + str(index) + '.png', dpi=300)
                plt.show()

                fig, ax1 = plt.subplots()
                ax1.set_xlabel('time step')
                ax1.set_ylabel('tactile reading')
                ax1.plot([None for i in range(9)] + [i for i in predicted_taxel_t10], alpha=0.5, c="b", label="t10")
                ax1.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel('loss')  # we already handled the x-label with ax1
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                fig.subplots_adjust(top=0.90)
                ax1.legend(loc="upper right")
                ax1.xaxis.set_major_locator(MultipleLocator(10))
                ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
                ax1.grid(which='minor')
                ax1.grid(which='major')
                plt.title("Simple_LSTM tactile " + str(index))
                plt.savefig(plot_save_dir + '/T10_test_plot_' + str(index) + '.png', dpi=300)
                plt.show()

    def load_scalars(self):
        self.scaler_tx = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/scaler_tx.pkl")
        self.scaler_ty = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/scaler_ty.pkl")
        self.scaler_tz = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/scaler_tz.pkl")
        self.min_max_scalerx_full_data = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/min_max_scalerx_full_data.pkl")
        self.min_max_scalery_full_data = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/min_max_scalery_full_data.pkl")
        self.min_max_scalerz_full_data = np.load("/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/min_max_scalerz_full_data.pkl")


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

if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTester()
    MT.test_full_model()
    MT.calc_test_performance()
    MT.create_test_plots(0)
    MT.create_test_plots(1)


