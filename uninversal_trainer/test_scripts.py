# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib
from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from pykalman import KalmanFilter

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

def scale_back(tactile_data, groundtruth_data, scaler_tx, scaler_ty, scaler_tz, min_max_scalerx_full_data, min_max_scalery_full_data, min_max_scalerz_full_data):
    pt_descalled_data = []
    gt_descalled_data = []

    for time_step in range(tactile_data.shape[0]):
        (ptx, pty, ptz) = np.split(tactile_data[time_step], 3, axis=1)
        (gtx, gty, gtz) = np.split(groundtruth_data[time_step], 3, axis=1)

        xela_ptx_inverse_minmax = min_max_scalerx_full_data.inverse_transform(ptx)
        xela_pty_inverse_minmax = min_max_scalery_full_data.inverse_transform(pty)
        xela_ptz_inverse_minmax = min_max_scalerz_full_data.inverse_transform(ptz)
        xela_ptx_inverse_full = scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
        xela_pty_inverse_full = scaler_ty.inverse_transform(xela_pty_inverse_minmax)
        xela_ptz_inverse_full = scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
        pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))

        xela_gtx_inverse_minmax = min_max_scalerx_full_data.inverse_transform(gtx)
        xela_gty_inverse_minmax = min_max_scalery_full_data.inverse_transform(gty)
        xela_gtz_inverse_minmax = min_max_scalerz_full_data.inverse_transform(gtz)
        xela_gtx_inverse_full = scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
        xela_gty_inverse_full = scaler_ty.inverse_transform(xela_gty_inverse_minmax)
        xela_gtz_inverse_full = scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
        gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))

    return np.array(pt_descalled_data), np.array(gt_descalled_data)



def all_plots():
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rcParams.update({'font.size': 22})

    # CDNA Plot:
    # folder_CDNA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_30_11_2021_15_24/universal_test_descaled/test_trial_"
    # trial = 0
    # taxel = 15
    # gt = np.load(folder_CDNA + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    # model_pt = np.load(folder_CDNA + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], model_pt[:, 0, taxel], c="g", label="CDNA" + " t+1", alpha=1.0)
    # plt.plot([i for i in range(len(model_pt))], model_pt[:, 4, taxel], c="b", label="CDNA" + " t+5", alpha=1.0)
    # plt.plot([i for i in range(len(model_pt))], model_pt[:, 9, taxel], c="k", label="CDNA" + " t+10", alpha=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()

    # # CDNA NEW Plot:
    # scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"
    # scaler_tx = load(open(scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
    # scaler_ty = load(open(scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
    # scaler_tz = load(open(scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
    # min_max_scalerx_full_data = load(open(scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
    # min_max_scalery_full_data = load(open(scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
    # min_max_scalerz_full_data = load(open(scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))
    # folder_CDNA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/20220223-111825-CDNA-8/test_pd_trial_10.npy"
    # folder_CDNA_GT = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/20220223-111825-CDNA-8/test_gt_trial_10.npy"
    # gt = np.load(folder_CDNA)
    # model_pt = np.load(folder_CDNA)
    #
    # gt, model_pt = scale_back(gt, model_pt, scaler_tx, scaler_ty, scaler_tz, min_max_scalerx_full_data, min_max_scalery_full_data, min_max_scalerz_full_data)
    #
    # for i in range(16,48):
    #     plt.plot([i for i in range(len(gt))], [gt[0, 0, i]] + list(gt[:-1, 0, i]), c="r", label="Ground truth", linewidth=1.0)
    #     plt.plot([i for i in range(len(model_pt))], model_pt[:, 1, i], c="k", label="CDNA" + " t+2", alpha=1.0)
    #     plt.plot([i for i in range(len(model_pt))], model_pt[:, 4, i], c="g", label="CDNA" + " t+6", alpha=1.0)
    #     plt.plot([i for i in range(len(model_pt))], model_pt[:, 9, i], c="b", label="CDNA" + " t+10", alpha=1.0)
    #     plt.grid(True)
    #     plt.title("Taxel sequence for trial: " + str(7))
    #     plt.xlabel('time step')
    #     plt.ylabel('tactile reading')
    #     plt.legend(loc="upper right")
    #     plt.show()

    # perfect plot:
    # folder_CDNA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_30_11_2021_15_24/universal_test_descaled/test_trial_"
    # trial = 0
    # taxel = 15
    # gt = np.load(folder_CDNA + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    # model_pt = np.load(folder_CDNA + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 0, taxel][1:]) + [None], c="g", label="Perfect Model" + " t+1", alpha=1.0)
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 4, taxel][4 + 1:]) + [None for i in range(4+1)], c="b", label="Perfect Model" + " t+5", alpha=1.0)
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel][9 + 1:]) + [None for i in range(9+1)], c="k", label="Perfect Model" + " t+10", alpha=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()

    # ACTVP and SVG:
    # folder_ACTVP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    # folder_SVG = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/test_trial_"
    # trial = 10
    # taxel = 6
    # gt = np.load(folder_ACTVP + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    #
    # model_pt = np.load(folder_ACTVP + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="ACTVP" + " t+10", alpha=1.0, linewidth=1.0)
    #
    # model_pt = np.load(folder_SVG + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="b", label="SVG" + " t+10", alpha=1.0, linewidth=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()

    # ACTVP and NEW WORKING SVG:
    # folder_ACTVP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    # folder_SVG = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_4layers_WORKING_model_14_02_2022_11_00/test_trial_"
    # trial = 10
    # taxel = 6
    # gt = np.load(folder_ACTVP + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    #
    # model_pt = np.load(folder_ACTVP + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="ACTVP" + " t+10", alpha=1.0, linewidth=1.0)
    #
    # model_pt = np.load(folder_SVG + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="b", label="SVG" + " t+10", alpha=1.0, linewidth=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()
    #

    # # PMN and PMN-AC
    # folder_PMN = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN/saved_models/box_only_WITHADD_model_17_12_2021_12_53/test_trial_"
    # folder_PMN_AC = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC/saved_models/box_only_dataset_WITH_ADD_model_14_12_2021_16_16/test_trial_"
    # trial = 10
    # taxel = 6
    # gt = np.load(folder_PMN + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    #
    # model_pt = np.load(folder_PMN + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="PMN" + " t+10", alpha=1.0, linewidth=1.0)
    #
    # model_pt = np.load(folder_PMN_AC + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="b", label="PMN-AC" + " t+10", alpha=1.0, linewidth=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()


    # # PMN-AC and PMN-AC-NA
    # folder_PMN_AC_NA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC_NA/saved_models/box_only_dataset_model_25_11_2021_14_10/test_trial_"
    # folder_PMN_AC = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC/saved_models/box_only_dataset_WITH_ADD_model_14_12_2021_16_16/test_trial_"
    # trial = 10
    # taxel = 6
    # gt = np.load(folder_PMN_AC + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    #
    # model_pt = np.load(folder_PMN_AC_NA + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="PMN-AC-NA" + " t+10", alpha=1.0, linewidth=1.0)
    #
    # model_pt = np.load(folder_PMN_AC + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="b", label="PMN-AC" + " t+10", alpha=1.0, linewidth=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()

    # ACTP VS ACTVP
    folder_ACTVP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    folder_ACTP_qual = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTP/saved_models/model_qual_12_01_2022_11_37/test_trial_"
    trial = 16
    for taxel in range(48):
    # taxel = 0
        gt = np.load(folder_ACTP_qual + str(trial) + "/groundtruth_data.npy")
        plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)

        model_pt = np.load(folder_ACTP_qual + str(trial) + "/prediction_data.npy")
        plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="ACTP" + " t+10", alpha=1.0, linewidth=1.0)

        model_pt = np.load(folder_ACTVP + str(trial) + "/prediction_data.npy")
        plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="b", label="ACTVP" + " t+10", alpha=1.0, linewidth=1.0)
        plt.grid(True)
        plt.title("Taxel  sequence for trial: " + str(trial))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="lower left")
        plt.show()

    # Intro Picture:
    # folder_ACTP_qual = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTP/saved_models/model_qual_12_01_2022_11_37/test_trial_"
    # trial = 16
    # taxel = 0
    # gt = np.load(folder_ACTP_qual + str(trial) + "/groundtruth_data.npy")
    # plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth", linewidth=1.0)
    #
    # model_pt = np.load(folder_ACTP_qual + str(trial) + "/prediction_data.npy")
    # plt.plot([i for i in range(len(model_pt))], list(model_pt[:, 9, taxel]), c="g", label="ACTP" + " t+10", alpha=1.0, linewidth=1.0)
    # plt.grid(True)
    # plt.title("Taxel  sequence for trial: " + str(trial))
    # plt.xlabel('time step')
    # plt.ylabel('tactile reading')
    # plt.legend(loc="lower left")
    # plt.show()


def compare_plots():
    # plot taxel GT's from different samples:
    folder_ACTVP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_04_01_2022_15_01/test_trial_"
    folder_ACTP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTP/saved_models/model_04_01_2022_16_05/test_trial_"
    folder_ACTP_qual = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTP/saved_models/model_qual_12_01_2022_11_37/test_trial_"
    folder_ACTVP_D = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/ACTVP/saved_models/model_double_07_01_2022_13_25/test_trial_"
    folder_CDNA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA/saved_models/model_30_11_2021_15_24/universal_test_descaled/test_trial_"
    folder_PMN = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN/saved_models/box_only_WITHADD_model_17_12_2021_12_53/test_trial_"
    folder_PMN_AC = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC/saved_models/box_only_dataset_WITH_ADD_model_14_12_2021_16_16/test_trial_"
    folder_PMN_AC_NA = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/PMN_AC_NA/saved_models/box_only_dataset_model_25_11_2021_14_10/test_trial_"
    # folder_SV2P = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SV2P/saved_models/box_only_model_14_12_2021_12_24/test_trial_"
    folder_SVG = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_3layers_model_13_12_2021_13_23/test_trial_"
    folder_MLP = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/MLP/saved_models/box_only_3layers_model_13_12_2021_13_23/test_trial_"
    folder_MLP_AC = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/MLP_AC/saved_models/box_only_model_09_12_2021_15_03/test_trial_"

    model_name  = ["folder_ACTP_qual"]  #, "SVG"]  # "ACTVP_D", "CDNA" "SV2P"
    folders     = [folder_ACTP_qual]  #folder_ACTVP, folder_PMN, folder_PMN_AC, folder_PMN_AC_NA]#, folder_SVG]  # folder_ACTVP_D, folder_CDNA, folder_SV2P
    trials      = [i for i in range(6, 22)]
    taxels      = [0] #  [i for i in range(48)]

    for trial in trials:
        for taxel, crol in zip(taxels, ["r", "g", "b"]):
            # plot ground truth:
            # gt = np.load(folders[0] + str(trial) + "/groundtruth_data.npy")
            # plt.plot([i for i in range(len(gt))], [None] + list(gt[:-1, 0, taxel]), c="g", label="Ground truth folder_ACTVP", linewidth=2.0)

            gt = np.load(folders[0] + str(trial) + "/groundtruth_data.npy")
            plt.plot([i for i in range(len(gt))], [gt[0, 0, taxel]] + list(gt[:-1, 0, taxel]), c=crol, label="Ground truth", linewidth=1.0)

            # gt = np.load(folders[2] + str(trial) + "/groundtruth_data.npy")
            # plt.plot([i for i in range(len(gt))], [None] + list(gt[:-1, 0, taxel]), c="r", label="Ground truth, folder_SVG", linewidth=2.0)

            for name, path, col in zip(model_name, folders, ["g", "b"]):
                meta_data = np.load(path + str(trial) + "/meta_data.npy")
                print(path + " : ", meta_data)
                model_pt = np.load(path + str(trial) + "/prediction_data.npy")

                # kf = KalmanFilter(initial_state_mean=model_pt[0, 9, taxel], transition_covariance=0.2, observation_covariance=400, n_dim_obs=1)
                # kf_smooth_fit = kf.em(model_pt[:, 9, taxel])
                # kf_filtered_data = kf_smooth_fit.filter(model_pt[:, 9, taxel])

                # plt.plot([i for i in range(len(model_pt) - 150)], model_pt[150:, 3, taxel], label=name + " t+4")
                # plt.plot([i for i in range(len(model_pt) - 150)], model_pt[150:, 6, taxel], label=name + " t+7")
                plt.plot([i for i in range(len(model_pt))], model_pt[:, 9, taxel], c=col, label=name + " t+10", alpha=1.0)
                # plt.plot([i for i in range(len(kf_filtered_data[0][:,0]))], kf_filtered_data[0][:,0], c="b", label=name + " t+10 kalman filtered", linewidth=1.5)

            plt.grid(True)
            plt.title("Taxel 18,19,20 sequence for trial: " + str(trial))  #  + " Taxel: " + str(taxel))
            plt.xlabel('time step')
            plt.ylabel('tactile reading')
            plt.legend(loc="lower left")
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


def checker():
    # ACTP TACTILE GT
    test_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_linear_qual_dataset_10c_10h_universal/"
    data_map = []
    with open(test_data_dir + 'map.csv', 'r') as f:  # rb
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)
    tactile_qual = np.load(test_data_dir + data_map[1][1])

    # ACTVP IMAGE GT
    test_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_image_dataset_10c_10h_32_universal/"
    data_map = []
    with open(test_data_dir + 'map.csv', 'r') as f:  # rb
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)
    tactile_image = np.load(test_data_dir + np.load(test_data_dir + data_map[1][2])[0])
    image_conv_back = cv2.resize(torch.tensor(tactile_image).numpy(), dsize=(4, 4),interpolation=cv2.INTER_CUBIC).flatten()

    # COMPARE ACTP TACTILE VS ACTVP IMAGE CONVERTED BACK
    print(tactile_qual[0][0])
    print(image_conv_back[0:16])


def create_plot_dataset():
    marker_state = np.array(pd.read_csv ("/home/user/Robotics/Data_sets/box_only_dataset/test/RSS_VIDEO_data_sample_2022-01-25-15-41-58/marker.csv", header=None))
    marker_data_x =  marker_state[1:,0].astype(float)
    marker_data_y =  marker_state[1:,1].astype(float)
    marker_data_z =  marker_state[1:,2].astype(float)

    GT = np.load("/home/user/Robotics/Data_sets/box_only_dataset/data2animate/GT_class.npy")
    PT = np.load("/home/user/Robotics/Data_sets/box_only_dataset/data2animate/t10_class.npy")
    # GT = GT[400:650]
    # PT = PT[400:650]
    fig = plt.figure()
    fig.set_size_inches(10, 2)
    ax = plt.axes(xlim=(0, len(GT)), ylim=(-0.1, 1.1))

    plt.title("Slip Classification on ACTP predicted data")
    # plt.title("ACTP model prediciton")
    plt.xlabel('Time step')
    plt.ylabel('Slip label (1 = slippage)')

    line1, = ax.plot([], [], lw=1, label="Groundtruth slip class")
    line2, = ax.plot([], [], lw=1, label="t+10 Prediction slip class")
    # line3, = ax.plot([], [], lw=1, label="Object Position Z")

    # ax.legend(loc="top left")

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        # line2.set_data([], [])
        return line1, line2, #line3,

    def animate(i):
        x = list([value for value in range(len(GT))])
        y1 = list ([value for value in GT[:i]] + [None for t in range(len(GT) - i)])
        y2 = list ([value for value in PT[:i]] + [None for t in range(len(PT) - i)])
        # y3 = list ([value for value in marker_data_z[:i]] + [None for t in range(len(marker_data_z) - i)])
        line1.set_data(x, y1)
        line2.set_data(x, y2)
        # line3.set_data(x, y3)
        return line1, line2, #line3,

    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(GT), interval=20.8, blit=True)
    ax.legend(loc="upper left")
    anim.save('/home/user/Robotics/Data_sets/box_only_dataset/slip_label.mp4', fps=48, extra_args=['-vcodec', 'libx264'], dpi=400)
    plt.show()

def create_image_plots():
    data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/RSS_VIDEO_test_linear_qual_dataset_10c_10h_universal/test_trial_33/"
    data_map = []
    with open(data_dir + 'map_33.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)

    robot_data   = []
    tactile_data = []
    for sequence in data_map[1:]:
        robot_data.append(np.load(data_dir + sequence[0]))
        tactile_data.append(np.load(data_dir + sequence[1]))
        print(sequence)

    images = []
    for frame in tactile_data:
        images.append(create_image(frame[0][0], frame[0][1], frame[0][2]))

    image_player(images=np.array(images), save_name="trial_33_rss", feature=2, experiment_to_test=33, data_save_path="/home/user/Robotics/Data_sets/box_only_dataset")
    # plot_player(plots=np.array(robot_data), save_name="trial_33_rss_ROBOT_", experiment_to_test=33, data_save_path="/home/user/Robotics/Data_sets/box_only_dataset")

    # robot_data = np.array(robot_data)
    #
    # for index, data in enumerate(robot_data):
    #     # fix euler x:
    #     if data[0, 3] > 0.5:
    #         robot_data[index, 0, 3] -= 0.5
    #
    #     elif data[0, 3] < 0.5:
    #         robot_data[index, 0, 3] += 0.5
    #
    #     # fix euler x:
    #     if data[0, 5] > 0.5:
    #         robot_data[index, 0, 5] -= 0.5
    #     elif data[0, 5] < 0.5:
    #         robot_data[index, 0, 5] += 0.5
    #
    # fig = plt.figure()
    # fig.set_size_inches(10, 4)
    # ax = plt.axes(xlim=(0, len(robot_data)), ylim=(0, 1))
    #
    # plt.title("Robot End Effector Orientation")
    # # plt.title("Taxel sequence for trial: 33")
    # plt.xlabel('Time step')
    # plt.ylabel('Orientation Normalised')
    #
    # line1, = ax.plot([], [], lw=1, label="Orientation x")
    # line2, = ax.plot([], [], lw=1, label="Orientation y")
    # line3, = ax.plot([], [], lw=1, label="Orientation z")
    # # line4, = ax.plot([], [], lw=1)
    # # line5, = ax.plot([], [], lw=1)
    # # line6, = ax.plot([], [], lw=1)
    #
    # ax.legend(loc="lower left")
    #
    # def init():
    #     line1.set_data([], [])
    #     line2.set_data([], [])
    #     line3.set_data([], [])
    #     # line4.set_data([], [])
    #     # line5.set_data([], [])
    #     # line6.set_data([], [])
    #     return line1, line2, line3 #, line4, line5, line6
    #
    # def animate(i):
    #     x = list([value for value in range(len(robot_data))])
    #     y1 = list ([value for value in robot_data[:i, 0, 3]] + [None for t in range(len(robot_data) - i)])
    #     y2 = list ([value for value in robot_data[:i, 0, 4]] + [None for t in range(len(robot_data) - i)])
    #     y3 = list ([value for value in robot_data[:i, 0, 5]] + [None for t in range(len(robot_data) - i)])
    #     # y4 = list ([value for value in robot_data[:i, 0, 3]] + [None for t in range(len(robot_data) - i)])
    #     # y5 = list ([value for value in robot_data[:i, 0, 4]] + [None for t in range(len(robot_data) - i)])
    #     # y6 = list ([value for value in robot_data[:i, 0, 5]] + [None for t in range(len(robot_data) - i)])
    #     line1.set_data(x, y1)
    #     line2.set_data(x, y2)
    #     line3.set_data(x, y3)
    #     # line4.set_data(x, y4)
    #     # line5.set_data(x, y5)
    #     # line6.set_data(x, y6)
    #     return line1, line2, line3#, line4, line5, line6
    #
    # anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(robot_data), interval=20.8, blit=True)
    # ax.legend(loc="lower left")
    # anim.save('/home/user/Robotics/Data_sets/box_only_dataset/orientation.mp4', fps=48, extra_args=['-vcodec', 'libx264'], dpi=400)
    # plt.show()

def create_image(tactile_x, tactile_y, tactile_z):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
            index += 1
    reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(4, 4), interpolation=cv2.INTER_CUBIC), k=3, axes=(0, 1))
    return reshaped_image


class plot_player():
    def __init__(self, plots, save_name, experiment_to_test, data_save_path):
        self.save_name = save_name
        self.experiment_to_test = experiment_to_test
        self.file_save_name = data_save_path + '/' + self.save_name + '.gif'
        print(self.file_save_name)
        self.run_the_tape(plots)

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rcParams.update({'font.size': 22})

    def init_plot(self):
        a = [i for i in range(100)], [i for i in range(100)]
        return a

    def update(self, i):
        plt.title(i)
        # self.im1.set_data(list([value for value in range(len(self.plots))]), list([value for value in self.plots[:self.indexyyy,0,0]] + [None for i in range(len(self.plots) - self.indexyyy)]))
        self.im1.set_data([t for t in range(100)], [t for t in range(100)])
        self.indexyyy += 1
        if self.indexyyy == len(self.plots):
            self.indexyyy = 0

    def run_the_tape(self, plots):
        self.indexyyy = 0
        self.plots = plots
        ax1 = plt.subplot(1, 2, 1)
        plt.title(0)
        plt.axes(xlim=(0, len(self.plots)), ylim=(-100, 100))
        plt.grid(True)
        plt.title("Taxel  sequence for trial: " + str(self.experiment_to_test))
        plt.xlabel('time step')
        plt.ylabel('tactile reading')
        plt.legend(loc="lower left")
        a, b = self.init_plot()
        print(len(a), len(b))
        self.im1 = ax1.plot(a, b, "-o", alpha=0.5, c="b", label="t1")
        self.im1 = self.im1[0]
        ani = FuncAnimation(plt.gcf(), self.update, interval=20.8, save_count=len(plots), repeat=False)
        # ani.save(self.file_save_name)
        ani.save(self.file_save_name, fps=48, extra_args=['-vcodec', 'libx264'], dpi=400)


class image_player():
    def __init__(self, images, save_name, feature, experiment_to_test, data_save_path):
        self.feature = feature
        self.save_name = save_name
        self.experiment_to_test = experiment_to_test
        self.file_save_name = data_save_path + '/' + self.save_name + '_feature_' + str(self.feature) + '.mp4'
        print(self.file_save_name)
        self.run_the_tape(images)

    def grab_frame(self):
        frame = self.images[self.indexyyy][:, :, self.feature] * 255
        return frame

    def update(self, i):
        plt.title(i)
        self.im1.set_data(self.grab_frame())
        self.indexyyy += 1
        if self.indexyyy == len(self.images):
            self.indexyyy = 0

    def run_the_tape(self, images):
        self.indexyyy = 0
        self.images = images
        ax1 = plt.subplot(1, 2, 1)
        self.im1 = ax1.imshow(self.grab_frame(), cmap='gray', vmin=0, vmax=255)
        ani = FuncAnimation(plt.gcf(), self.update, interval=20.8, save_count=len(images), repeat=False)
        ani.save(self.file_save_name, fps=48, extra_args=['-vcodec', 'libx264'], dpi=400)


def create_scaled_plots__():
    data_path = "/home/user/Robotics/tactile_prediction/Plots_for_kiyanoush/PMN-AC/box_only_dataset_model_NO_ADD_25_11_2021_14_10/"
    save_dir = "/home/user/Robotics/tactile_prediction/Plots_for_kiyanoush/PMN-AC-NA/"
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

    scaler_tx = load (open (scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
    scaler_ty = load (open (scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
    scaler_tz = load (open (scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
    min_max_scalerx_full_data = load (open (scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
    min_max_scalery_full_data = load (open (scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
    min_max_scalerz_full_data = load (open (scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

    for trial_number in range(22):
        pd = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/prediction_data_t10.npy")
        gt = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy")
        meta = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/meta_data.npy")

        pt_descalled_data = []
        gt_descalled_data = []
        (ptx, pty, ptz) = np.split(pd, 3, axis=1)
        (gtx, gty, gtz) = np.split(gt, 3, axis=1)
        for time_step in range(pd.shape[0]):
            xela_ptx_inverse_minmax = scaler_tx.transform(ptx[time_step].reshape(1, -1))
            xela_pty_inverse_minmax = scaler_ty.transform(pty[time_step].reshape(1, -1))
            xela_ptz_inverse_minmax = scaler_tz.transform(ptz[time_step].reshape(1, -1))
            xela_ptx_inverse_full = min_max_scalerx_full_data.transform(xela_ptx_inverse_minmax)
            xela_pty_inverse_full = min_max_scalery_full_data.transform(xela_pty_inverse_minmax)
            xela_ptz_inverse_full = min_max_scalerz_full_data.transform(xela_ptz_inverse_minmax)
            pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1).squeeze())

            xela_gtx_inverse_minmax = scaler_tx.transform(gtx[time_step].reshape(1, -1))
            xela_gty_inverse_minmax = scaler_ty.transform(gty[time_step].reshape(1, -1))
            xela_gtz_inverse_minmax = scaler_tz.transform(gtz[time_step].reshape(1, -1))
            xela_gtx_inverse_full = min_max_scalerx_full_data.transform(xela_gtx_inverse_minmax)
            xela_gty_inverse_full = min_max_scalery_full_data.transform(xela_gty_inverse_minmax)
            xela_gtz_inverse_full = min_max_scalerz_full_data.transform(xela_gtz_inverse_minmax)
            gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1).squeeze())

        path_save = save_dir + "SCALED_test_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data_t10", np.array(pt_descalled_data))
        np.save(path_save + "groundtruth_data", np.array(gt_descalled_data))
        np.save(path_save + "meta_data", meta)

    for trial_number in range(51):
        pd = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/prediction_data_t10.npy")
        gt = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy")
        meta = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/meta_data.npy")

        pt_descalled_data = []
        gt_descalled_data = []
        (ptx, pty, ptz) = np.split(pd, 3, axis=1)
        (gtx, gty, gtz) = np.split(gt, 3, axis=1)
        for time_step in range(pd.shape[0]):
            xela_ptx_inverse_minmax = scaler_tx.transform(ptx[time_step].reshape(1, -1))
            xela_pty_inverse_minmax = scaler_ty.transform(pty[time_step].reshape(1, -1))
            xela_ptz_inverse_minmax = scaler_tz.transform(ptz[time_step].reshape(1, -1))
            xela_ptx_inverse_full = min_max_scalerx_full_data.transform(xela_ptx_inverse_minmax)
            xela_pty_inverse_full = min_max_scalery_full_data.transform(xela_pty_inverse_minmax)
            xela_ptz_inverse_full = min_max_scalerz_full_data.transform(xela_ptz_inverse_minmax)
            pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1).squeeze())

            xela_gtx_inverse_minmax = scaler_t.transform(gtx[time_step].reshape(1, -1))
            xela_gty_inverse_minmax = scaler_t.transform(gty[time_step].reshape(1, -1))
            xela_gtz_inverse_minmax = scaler_t.transform(gtz[time_step].reshape(1, -1))
            xela_gtx_inverse_full = xmin_max_scalerx_full_data.transform(xela_gtx_inverse_minmax)
            xela_gty_inverse_full = ymin_max_scalery_full_data.transform(xela_gty_inverse_minmax)
            xela_gtz_inverse_full = zmin_max_scalerz_full_data.transform(xela_gtz_inverse_minmax)
            gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1).squeeze())

        path_save = save_dir + "SCALED_train_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data_t10", np.array(pt_descalled_data))
        np.save(path_save + "groundtruth_data", np.array(gt_descalled_data))
        np.save(path_save + "meta_data", meta)

def create_scaled_plots():
    data_path = "/home/user/Robotics/tactile_prediction/Plots_for_kiyanoush/PMN-AC/box_only_dataset_model_NO_ADD_25_11_2021_14_10/"
    save_dir = "/home/user/Robotics/tactile_prediction/Plots_for_kiyanoush/PMN-AC-NA/"
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

    scaler_tx = load (open (scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
    scaler_ty = load (open (scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
    scaler_tz = load (open (scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
    min_max_scalerx_full_data = load (open (scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
    min_max_scalery_full_data = load (open (scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
    min_max_scalerz_full_data = load (open (scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

    full_gt_data = []

    for trial_number in range(22):
        print(trial_number)
        full_gt_data += list(np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy"))

    for trial_number in range(51):
        print(trial_number)
        full_gt_data += list(np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy"))


    full_gt_data = np.array(full_gt_data)
    (ptx, pty, ptz) = np.split(full_gt_data, 3, axis=1)
    full_gt_data = np.concatenate((np.expand_dims(ptx, axis=1), np.expand_dims(pty, axis=1), np.expand_dims(ptz, axis=1)), axis=1)

    tactile_standard_scaler = [preprocessing.StandardScaler().fit(full_gt_data[:, feature]) for feature in range(3)]
    tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tactile_standard_scaler[feature].transform(full_gt_data[:, feature])) for feature in range(3)]


    for trial_number in range(22):
        pd = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/prediction_data_t10.npy")
        gt = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy")
        meta = np.load(data_path + "test_SCALED_test_plots_" + str(trial_number) + "/meta_data.npy")

        (ptx, pty, ptz) = np.split(pd, 3, axis=1)
        (gtx, gty, gtz) = np.split(gt, 3, axis=1)

        pt_data = [ptx, pty, ptz]
        gt_data = [gtx, gty, gtz]

        for index, (standard_scaler, min_max_scalar) in enumerate(zip(tactile_standard_scaler, tactile_min_max_scalar)):
            pt_data[index] = standard_scaler.transform(pt_data[index])
            pt_data[index] = min_max_scalar.transform(pt_data[index])
            gt_data[index] = standard_scaler.transform(gt_data[index])
            gt_data[index] = min_max_scalar.transform(gt_data[index])


        path_save = save_dir + "SCALED_test_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data_t10", np.array(pt_descalled_data))
        np.save(path_save + "groundtruth_data", np.array(gt_descalled_data))
        np.save(path_save + "meta_data", meta)

    for trial_number in range(51):
        pd = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/prediction_data_t10.npy")
        gt = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/trial_groundtruth_data.npy")
        meta = np.load(data_path + "train_SCALED_test_plots_" + str(trial_number) + "/meta_data.npy")



        path_save = save_dir + "SCALED_train_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data_t10", np.array(pt_descalled_data))
        np.save(path_save + "groundtruth_data", np.array(gt_descalled_data))
        np.save(path_save + "meta_data", meta)



def main():
    # compare_plots()
    # create_image_plots()
    # all_plots()
    create_scaled_plots()
    # create_plot_dataset()


    # create_loss()
    # plot_perfect()
    # plot_aligned()
    # checker()
# tactile_data_sequence_0.npy

if __name__ == "__main__":
    main()