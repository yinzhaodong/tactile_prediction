# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

data_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/PixelMotionNet/saved_models/box_only_model_29_11_2021_16_58/"
# print (np.load (data_save_path + "plot_training_loss.npy"))
# print (np.load (data_save_path + "plot_validation_loss.npy"))
print (np.load (data_save_path + "model_performance_loss_data.npy"))
