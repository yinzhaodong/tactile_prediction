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


a = torch.ones([4,4,3])*0.7
b = torch.ones([4,4,3])*0.2
print(a + b)
# torch.sum(torch.tensor([a, b]))