# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
# This script is used to convert the tactile dataset into a set of images.
# 	- It saves a map.csv file that points to all the correct images in the sequence of data.
#		-  Tactile sequence, d 1*48 -> 64*64*3 int8, [folder_name/image_name_0, folder_name/image_name_1, ..., folder_name/image_name_N]
#		-  Robot sequence, r 1*6 float16, [[x0,y0,z0,r0,p0,y0], [x1,y1,z1,r1,p1,y1], ..., [xN,yN,zN,rN,pN,yN]]
#		-  Slip Classification, s bool, [s0, s1, ..., sN]

import os
import cv2
import csv
import glob
import click
import logging
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pickle import dump
from dotenv import find_dotenv, load_dotenv
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import map_coordinates

image_height, image_width = 32, 32


def create_image(tactile_x, tactile_y, tactile_z):
	# convert tactile data into an image:
	image = np.zeros((4,4,3), np.float32)
	index = 0
	for x in range(4):
		for y in range(4):
			image[x][y] =  [tactile_x[index],
							tactile_y[index],
							tactile_z[index]]
			index += 1
	reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
	return reshaped_image


def create_taxels(image_to_convert):
	reshaped_image = np.rot90(cv2.resize(image_to_convert.astype(np.float32), dsize=(4, 4), interpolation=cv2.INTER_CUBIC), k=3, axes=(0, 1))
	reshaped_image = reshaped_image.flatten()
	tactile_x = []
	tactile_y = []
	tactile_z = []
	for index in range(0, 48, 3):
		tactile_x.append(reshaped_image[index])
		tactile_y.append(reshaped_image[index+1])
		tactile_z.append(reshaped_image[index+2])

	return tactile_x + tactile_y + tactile_z


if __name__ == "__main__":
	tactile_x = [0.01 * i for i in range(0, 16)]
	tactile_y = [1.0 * i for i in range(0, 16)]
	tactile_z = [100.0 * i for i in range(0, 16)]

	tactile = tactile_x + tactile_y + tactile_z
	print("Target: ", tactile)

	image = create_image(tactile_x, tactile_y, tactile_z)
	# print("image: ", image)

	tactile_reconstructed = create_taxels(image)
	print([round(i, 3) for i in tactile_reconstructed])

