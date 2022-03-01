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
    SVG_folder = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/test_descaled/test_DESCALED_test_plots_"
    # folder_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_01/universal_test_descaled/test_trial_"

    for trial in range(23):
        SVG_meta = np.load(SVG_folder + str(trial) + "/meta_data.npy")
        print("SVG_meta: ", SVG_meta)

        SVG_gt = np.load(SVG_folder + str(trial) + "/groundtruth_data.npy")
        SVG_pt = np.load(SVG_folder + str(trial) + "/prediction_data.npy")

        plt.plot([i for i in range(len(SVG_gt))], [None] + list(SVG_gt[:-1, 0, 25]), c="r", label="Ground truth")

        plt.plot([i for i in range(len(SVG_pt))], list(SVG_pt[3:, 3, 25]) + [None for i in range(3)], label="Perfect t+3")
        plt.plot([i for i in range(len(SVG_pt))], list(SVG_pt[6:, 6, 25]) + [None for i in range(6)], label="Perfect t+6")
        plt.plot([i for i in range(len(SVG_pt))], list(SVG_pt[9:, 9, 25]) + [None for i in range(9)], c="g", label="Perfect t+10")

        plt.grid(True)
        plt.title("SVG - Trial: " + str(trial))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="upper right")
        plt.show()


def main():
    plot_aligned()


if __name__ == "__main__":
    main()
