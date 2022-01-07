# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


def plot_aligned():
    folder_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_01/universal_test_descaled/test_trial_"

    for trial in range(23):
        ACTP_meta = np.load(folder_2 + str(trial) + "/meta_data.npy")

        print("CDNA: ", ACTP_meta)

        ACTP_gt = np.load(folder_2 + str(trial) + "/groundtruth_data.npy")
        ACTP_pt = np.load(folder_2 + str(trial) + "/prediction_data.npy")

        plt.plot([i for i in range(len(ACTP_gt))], [None] + list(ACTP_gt[:-1, 0, 25]), c="r", label="Ground truth")

        # plt.plot([i for i in range(len(ACTP_pt))], ACTP_pt[:, 0, 25], label="CDNA t+0")
        plt.plot([i for i in range(len(ACTP_pt))], ACTP_pt[:, 3, 25], label="CDNA t+3")
        plt.plot([i for i in range(len(ACTP_pt))], ACTP_pt[:, 6, 25], label="CDNA t+6")
        plt.plot([i for i in range(len(ACTP_pt))], ACTP_pt[:, 9, 25], c="g", label="CDNA t+10")

        plt.grid(True)
        plt.title("CDNA Predictions - Trial: " + str(trial))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="upper right")
        plt.show()

def plot_perfect():
    folder_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_01/universal_test_descaled/test_trial_"

    for trial in range(23):
        ACTP_meta = np.load(folder_2 + str(trial) + "/meta_data.npy")

        print("CDNA: ", ACTP_meta)

        ACTP_gt = np.load(folder_2 + str(trial) + "/groundtruth_data.npy")
        ACTP_pt = np.load(folder_2 + str(trial) + "/prediction_data.npy")

        plt.plot([i for i in range(len(ACTP_gt))], [None] + list(ACTP_gt[:-1, 0, 25]), c="r", label="Ground truth")

        # plt.plot([i for i in range(len(ACTP_pt))], ACTP_pt[:, 0, 25], label="CDNA t+0")
        plt.plot([i for i in range(len(ACTP_pt))], list(ACTP_pt[3:, 3, 25]) + [None for i in range(3)], label="Perfect t+3")
        plt.plot([i for i in range(len(ACTP_pt))], list(ACTP_pt[6:, 6, 25]) + [None for i in range(6)], label="Perfect t+6")
        plt.plot([i for i in range(len(ACTP_pt))], list(ACTP_pt[9:, 9, 25]) + [None for i in range(9)], c="g", label="Perfect t+10")

        plt.grid(True)
        plt.title("Perfect Prediction - Trial: " + str(trial))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="upper right")
        plt.show()


def compare_plots():
    # plot taxel GT's from different samples:
    folder_ACTVP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    folder_CDNA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_30_11_2021_15_24/universal_test_descaled/test_trial_"
    folder_PMN = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN/saved_models/box_only_WITHADD_model_17_12_2021_12_53/test_trial_"
    folder_PMN_AC = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC/saved_models/box_only_dataset_WITH_ADD_model_14_12_2021_16_16/test_trial_"
    folder_PMN_AC_NA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC_NA/saved_models/box_only_dataset_model_25_11_2021_14_10/test_trial_"
    folder_SV2P = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SV2P/saved_models/box_only_model_14_12_2021_12_24/test_trial_"
    folder_SVG = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/test_trial_"

    model_name = ["ACTVP", "CDNA", "PMN", "PMN_AC", "PMN_AC_NA", "SV2P", "SVG"]
    folders = [folder_ACTVP, folder_CDNA, folder_PMN, folder_PMN_AC, folder_PMN_AC_NA, folder_SV2P, folder_SVG]
    trials  = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    taxel = 11

    for trial in trials:
        # plot ground truth:
        gt = np.load(folders[0] + str(trial) + "/groundtruth_data.npy")
        plt.plot([i for i in range(len(gt))], [None] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth")

        for name, path in zip(model_name, folders):
            meta_data = np.load(path + str(trial) + "/meta_data.npy")
            print(path + " : ", meta_data)
            model_pt = np.load(path + str(trial) + "/prediction_data.npy")
            plt.plot([i for i in range(len(model_pt))], model_pt[:, 9, taxel], label=name + " t+10")

        plt.grid(True)
        plt.title("Model Predictions - Trial: " + str(trial) + " Taxel: " + str(taxel))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="upper right")
        plt.show()


def create_loss():
    # plot taxel GT's from different samples:
    folder_1 = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    folder_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_01/universal_test_descaled/test_trial_"

    actp_loss = 0.0
    actvp_loss =0.0

    for trial in range(23):
        ACTP_meta = np.load(folder_2 + str(trial) + "/meta_data.npy")
        ACTVP_meta = np.load(folder_1 + str(trial) + "/meta_data.npy")

        ACTP_gt = np.load(folder_2 + str(trial) + "/groundtruth_data.npy")
        ACTP_pt = np.load(folder_2 + str(trial) + "/prediction_data.npy")

        ACTVP_gt = np.load(folder_1 + str(trial) + "/groundtruth_data.npy")
        ACTVP_pt = np.load(folder_1 + str(trial) + "/prediction_data.npy")

        mae_loss = 0.0
        criterion = nn.L1Loss()
        for index, (gt, pt) in enumerate(zip(ACTP_gt, ACTP_pt)):
            mae_loss += criterion(torch.tensor(gt), torch.tensor(pt)).item()
        actp_loss += mae_loss / (index+1)

        mae_loss = 0.0
        criterion = nn.L1Loss ()
        for index, (gt, pt) in enumerate(zip(ACTVP_gt, ACTVP_pt)):
            mae_loss += criterion(torch.tensor(gt), torch.tensor(pt)).item()
        actvp_loss += mae_loss / (index+1)

    print(actp_loss / 23)
    print(actvp_loss / 23)


def main():
    # create_loss()
    # plot_perfect()
    # plot_aligned()
    compare_plots()


if __name__ == "__main__":
    main()