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
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import map_coordinates

# Hyperparameters:
data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_001_122/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/manual_slip_detection/'
context_length = 10
horrizon_length = 10
image_height, image_width = 64, 64

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

# read all the folder names for each trial
files = glob.glob(data_dir + '/*')
path_file = []
index_to_save = 0


for experiment_number in tqdm(range(len(files))):
	robot_state  = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
	xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))
	meta_data = np.asarray(pd.read_csv(files[experiment_number] + '/meta_data.csv', header=None))

	####################################### Robot Data ###########################################
	ee_positions = []
	ee_position_x, ee_position_y, ee_position_z = [], [], []
	ee_orientation_x, ee_orientation_y, ee_orientation_z = [], [], []
	ee_orientation_quat_x, ee_orientation_quat_y, ee_orientation_quat_z, ee_orientation_quat_w = [], [], [], []

	for state in robot_state[1:]:
		ee_positions.append([float(item) for item in robot_state[1][-7:-4]])
		ee_position_x.append(state[-7])
		ee_position_y.append(state[-6])
		ee_position_z.append(state[-5])
		# quat
		ee_orientation_quat_x.append(state[-4])
		ee_orientation_quat_y.append(state[-3])
		ee_orientation_quat_z.append(state[-2])
		ee_orientation_quat_w.append(state[-1])
		# euler
		ee_orientation = R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)
		ee_orientation_x.append(ee_orientation[0])
		ee_orientation_y.append(ee_orientation[1])
		ee_orientation_z.append(ee_orientation[2])

	ee_position_x = np.asarray(ee_position_x).astype(float)
	ee_position_y = np.asarray(ee_position_y).astype(float)
	ee_position_z = np.asarray(ee_position_z).astype(float)
	# quat:
	ee_orientation_quat_x = np.asarray(ee_orientation_quat_x).astype(float)
	ee_orientation_quat_y = np.asarray(ee_orientation_quat_y).astype(float)
	ee_orientation_quat_z = np.asarray(ee_orientation_quat_z).astype(float)
	ee_orientation_quat_w = np.asarray(ee_orientation_quat_w).astype(float)
	# euler:
	ee_orientation_x = np.asarray(ee_orientation_x).astype(float)
	ee_orientation_y = np.asarray(ee_orientation_y).astype(float)
	ee_orientation_z = np.asarray(ee_orientation_z).astype(float)

	# normalise for each value:
	min_x_position_x, max_x_position_x = (min(ee_position_x), max(ee_position_x))
	min_y_position_y, max_y_position_y = (min(ee_position_y), max(ee_position_y))
	min_z_position_z, max_z_position_z = (min(ee_position_z), max(ee_position_z))
	#quat
	min_x_orientation_quat_x, max_x_orientation_quat_x = (min(ee_orientation_quat_x), max(ee_orientation_quat_x))
	min_y_orientation_quat_y, max_y_orientation_quat_y = (min(ee_orientation_quat_y), max(ee_orientation_quat_y))
	min_z_orientation_quat_z, max_z_orientation_quat_z = (min(ee_orientation_quat_z), max(ee_orientation_quat_z))
	min_z_orientation_quat_w, max_z_orientation_quat_w = (min(ee_orientation_quat_w), max(ee_orientation_quat_w))
	# euler:
	min_x_orientation_x, max_x_orientation_x = (min(ee_orientation_x), max(ee_orientation_x))
	min_y_orientation_y, max_y_orientation_y = (min(ee_orientation_y), max(ee_orientation_y))
	min_z_orientation_z, max_z_orientation_z = (min(ee_orientation_z), max(ee_orientation_z))

	for time_step in range(len(ee_position_x)):
		ee_position_x[time_step] = (ee_position_x[time_step] - min_x_position_x) / (max_x_position_x - min_x_position_x) 
		ee_position_y[time_step] = (ee_position_y[time_step] - min_y_position_y) / (max_y_position_y - min_y_position_y) 
		ee_position_z[time_step] = (ee_position_z[time_step] - min_z_position_z) / (max_z_position_z - min_z_position_z)
		# quat:
		ee_orientation_quat_x[time_step] = (ee_orientation_quat_x[time_step] - min_x_orientation_quat_x) / (max_x_orientation_quat_x - min_x_orientation_quat_x) 
		ee_orientation_quat_y[time_step] = (ee_orientation_quat_y[time_step] - min_y_orientation_quat_y) / (max_y_orientation_quat_y - min_y_orientation_quat_y) 
		ee_orientation_quat_z[time_step] = (ee_orientation_quat_z[time_step] - min_z_orientation_quat_z) / (max_z_orientation_quat_z - min_z_orientation_quat_z)
		ee_orientation_quat_w[time_step] = (ee_orientation_quat_w[time_step] - min_z_orientation_quat_w) / (max_z_orientation_quat_w - min_z_orientation_quat_w)
		# euler:
		ee_orientation_x[time_step] = (ee_orientation_x[time_step] - min_x_orientation_x) / (max_x_orientation_x - min_x_orientation_x) 
		ee_orientation_y[time_step] = (ee_orientation_y[time_step] - min_y_orientation_y) / (max_y_orientation_y - min_y_orientation_y) 
		ee_orientation_z[time_step] = (ee_orientation_z[time_step] - min_z_orientation_z) / (max_z_orientation_z - min_z_orientation_z)

	ee_position_x = np.asarray(ee_position_x)
	ee_position_y = np.asarray(ee_position_y)
	ee_position_z = np.asarray(ee_position_z)
	# quat:
	ee_orientation_quat_x = np.asarray(ee_orientation_quat_x)
	ee_orientation_quat_y = np.asarray(ee_orientation_quat_y)
	ee_orientation_quat_z = np.asarray(ee_orientation_quat_z)
	ee_orientation_quat_w = np.asarray(ee_orientation_quat_w)
	# euler:
	ee_orientation_x = np.asarray(ee_orientation_x)
	ee_orientation_y = np.asarray(ee_orientation_y)
	ee_orientation_z = np.asarray(ee_orientation_z)


	####################################### Xela Sensor Data ###########################################
	xela_sensor1_data_x, xela_sensor1_data_y, xela_sensor1_data_z = [], [], []
	xela_sensor2_data_x, xela_sensor2_data_y, xela_sensor2_data_z = [], [], []
	xela_sensor1_data_x_mean, xela_sensor1_data_y_mean, xela_sensor1_data_z_mean = [], [], []
	xela_sensor2_data_x_mean, xela_sensor2_data_y_mean, xela_sensor2_data_z_mean = [], [], []

	for sample1, sample2 in zip(xela_sensor1[1:], xela_sensor2[1:]):
		sample1_data_x, sample1_data_y, sample1_data_z = [], [], []
		sample2_data_x, sample2_data_y, sample2_data_z = [], [], []

		for i in range(0, len(xela_sensor1[0]), 3):
			sample1_data_x.append(float(sample1[i]))
			sample1_data_y.append(float(sample1[i+1]))
			sample1_data_z.append(float(sample1[i+2]))

			# sample2_data_x.append(float(sample2[i]))
			# sample2_data_y.append(float(sample2[i+1]))
			# sample2_data_z.append(float(sample2[i+2]))

		xela_sensor1_data_x.append(sample1_data_x)
		xela_sensor1_data_y.append(sample1_data_y)
		xela_sensor1_data_z.append(sample1_data_z)

		# xela_sensor2_data_x.append(sample2_data_x)
		# xela_sensor2_data_y.append(sample2_data_y)
		# xela_sensor2_data_z.append(sample2_data_z)

	# mean starting values:
	xela_sensor1_average_starting_value_x = int(sum(xela_sensor1_data_x[0]) / len(xela_sensor1_data_x[0]))
	xela_sensor1_average_starting_value_y = int(sum(xela_sensor1_data_y[0]) / len(xela_sensor1_data_y[0]))
	xela_sensor1_average_starting_value_z = int(sum(xela_sensor1_data_z[0]) / len(xela_sensor1_data_z[0]))
	xela_sensor1_offset_x = [xela_sensor1_average_starting_value_x - tactile_starting_value for tactile_starting_value in xela_sensor1_data_x[0]]
	xela_sensor1_offset_y = [xela_sensor1_average_starting_value_y - tactile_starting_value for tactile_starting_value in xela_sensor1_data_y[0]]
	xela_sensor1_offset_z = [xela_sensor1_average_starting_value_z - tactile_starting_value for tactile_starting_value in xela_sensor1_data_z[0]]
	# xela_sensor2_average_starting_value_x = int(sum(xela_sensor2_data_x[0]) / len(xela_sensor2_data_x[0]))
	# xela_sensor2_average_starting_value_y = int(sum(xela_sensor2_data_y[0]) / len(xela_sensor2_data_y[0]))
	# xela_sensor2_average_starting_value_z = int(sum(xela_sensor2_data_z[0]) / len(xela_sensor2_data_z[0]))
	# xela_sensor2_offset_x = [xela_sensor2_average_starting_value_x - tactile_starting_value for tactile_starting_value in xela_sensor2_data_x[0]]
	# xela_sensor2_offset_y = [xela_sensor2_average_starting_value_y - tactile_starting_value for tactile_starting_value in xela_sensor2_data_y[0]]
	# xela_sensor2_offset_z = [xela_sensor2_average_starting_value_z - tactile_starting_value for tactile_starting_value in xela_sensor2_data_z[0]]

	# normalise for each force:
	min_x_sensor1, max_x_sensor1 = (min([min(x) for x in xela_sensor1_data_x]), max([max(x) for x in xela_sensor1_data_x]))
	min_y_sensor1, max_y_sensor1 = (min([min(y) for y in xela_sensor1_data_y]), max([max(y) for y in xela_sensor1_data_y]))
	min_z_sensor1, max_z_sensor1 = (min([min(z) for z in xela_sensor1_data_z]), max([max(z) for z in xela_sensor1_data_z]))
	# min_x_sensor2, max_x_sensor2 = (min([min(x) for x in xela_sensor2_data_x]), max([max(x) for x in xela_sensor2_data_x]))
	# min_y_sensor2, max_y_sensor2 = (min([min(y) for y in xela_sensor2_data_y]), max([max(y) for y in xela_sensor2_data_y]))
	# min_z_sensor2, max_z_sensor2 = (min([min(z) for z in xela_sensor2_data_z]), max([max(z) for z in xela_sensor2_data_z]))

	for time_step in range(len(xela_sensor1_data_x)):
		xela_sensor2_sample_x_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_x, xela_sensor1_data_x[time_step])]
		xela_sensor2_sample_y_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_y, xela_sensor1_data_y[time_step])]
		xela_sensor2_sample_z_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_z, xela_sensor1_data_z[time_step])]
		# xela_sensor2_sample_x_test = [offset+real_value for offset, real_value in zip(xela_sensor2_offset_x, xela_sensor2_data_x[time_step])]
		# xela_sensor2_sample_y_test = [offset+real_value for offset, real_value in zip(xela_sensor2_offset_y, xela_sensor2_data_y[time_step])]
		# xela_sensor2_sample_z_test = [offset+real_value for offset, real_value in zip(xela_sensor2_offset_z, xela_sensor2_data_z[time_step])]
		for i in range(np.asarray(xela_sensor1_data_x).shape[1]):
			xela_sensor1_data_x[time_step][i] = (xela_sensor2_sample_x_test[i] - min_x_sensor1) / (max_x_sensor1 - min_x_sensor1) 
			xela_sensor1_data_y[time_step][i] = (xela_sensor2_sample_y_test[i] - min_y_sensor1) / (max_y_sensor1 - min_y_sensor1) 
			xela_sensor1_data_z[time_step][i] = (xela_sensor2_sample_z_test[i] - min_z_sensor1) / (max_z_sensor1 - min_z_sensor1)
			# xela_sensor2_data_x[time_step][i] = (xela_sensor2_sample_x_test[i] - min_x_sensor2) / (max_x_sensor2 - min_x_sensor2) 
			# xela_sensor2_data_y[time_step][i] = (xela_sensor2_sample_y_test[i] - min_y_sensor2) / (max_y_sensor2 - min_y_sensor2) 
			# xela_sensor2_data_z[time_step][i] = (xela_sensor2_sample_z_test[i] - min_z_sensor2) / (max_z_sensor2 - min_z_sensor2)


	# create images from xela_data
	if SAVE_IMAGES == True:
		xela_images_1, xela_images_2 = [], []
		for time_step in range(len(xela_sensor1_data_x)):
			xela_images_1.append(create_image(xela_sensor1_data_x[time_step], xela_sensor1_data_y[time_step], xela_sensor1_data_z[time_step]))
			# xela_images_2.append(create_image(xela_sensor2_data_x[time_step], xela_sensor2_data_y[time_step], xela_sensor2_data_z[time_step]))

	####################################### Format data into time series ###########################################
	for sample in range(0, len(ee_position_x) - sequence_length):
		robot_data_euler_sequence, robot_data_quat_sequence, xela_1_sequence_data, xela_2_sequence_data, experiment_data_sequence, time_step_data_sequence, xela_image_1_data_sequence, xela_image_2_data_sequence = [], [], [], [], [], [], [], []
		for t in range(0, sequence_length):
			robot_data_euler_sequence.append([ee_position_x[sample+t], ee_position_y[sample+t], ee_position_z[sample+t], ee_orientation_x[sample+t], ee_orientation_y[sample+t], ee_orientation_z[sample+t]])
			robot_data_quat_sequence.append([ee_position_x[sample+t], ee_position_y[sample+t], ee_position_z[sample+t], ee_orientation_quat_x[sample+t], ee_orientation_quat_y[sample+t], ee_orientation_quat_z[sample+t], ee_orientation_quat_w[sample+t]])
			xela_1_sequence_data.append(np.column_stack((xela_sensor1_data_x[sample+t], xela_sensor1_data_y[sample+t], xela_sensor1_data_z[sample+t])).flatten())
			# xela_2_sequence_data.append(np.column_stack((xela_sensor2_data_x[sample+t], xela_sensor2_data_y[sample+t], xela_sensor2_data_z[sample+t])).flatten())
			if SAVE_IMAGES == True:
				xela_image_1_data_sequence.append(xela_images_1[sample+t])
				# xela_image_2_data_sequence.append(xela_images_2[sample+t])
			experiment_data_sequence.append(experiment_number)
			time_step_data_sequence.append(sample+t)
		# robot_data.append(robot_data_sequence)
		# xela_1_data.append(xela_1_sequence_data)
		# xela_2_data.append(xela_2_sequence_data)
		# xela_image_1_data.append(xela_image_1_data_sequence)
		# xela_image_2_data.append(xela_image_2_data_sequence)
		# experiment_data.append(experiment_number)
		# time_step_data.append(time_step_data_sequence)

		np.save(out_dir + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
		np.save(out_dir + 'robot_data_quat_' + str(index_to_save), robot_data_quat_sequence)
		np.save(out_dir + 'xela_1_data_' + str(index_to_save), xela_1_sequence_data)
		np.save(out_dir + 'xela_2_data_' + str(index_to_save), xela_2_sequence_data)
		if SAVE_IMAGES == True:
			np.save(out_dir + 'xela_1_image_data_' + str(index_to_save), xela_image_1_data_sequence)
			# np.save(out_dir + 'xela_2_image_data_' + str(index_to_save), xela_image_2_data_sequence)
		np.save(out_dir + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
		np.save(out_dir + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
		ref = []
		ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
		ref.append('robot_data_quat_' + str(index_to_save) + '.npy')
		ref.append('xela_1_data_' + str(index_to_save) + '.npy')
		ref.append('xela_2_data_' + str(index_to_save) + '.npy')
		if SAVE_IMAGES == True:
			ref.append('xela_1_image_data_' + str(index_to_save) + '.npy')
			# ref.append('xela_2_image_data_' + str(index_to_save) + '.npy')
		ref.append('experiment_number_' + str(index_to_save) + '.npy')
		ref.append('time_step_data_' + str(index_to_save) + '.npy')
		path_file.append(ref)
		index_to_save += 1

with open(out_dir + '/map.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	if SAVE_IMAGES == True:
		writer.writerow(['robot_data_path_euler', 'robot_data_path_quat', 'xela_1_data_path', 'xela_2_data_path', 'xela_1_image_data_path', 'experiment_number', 'time_steps'])
	if SAVE_IMAGES == False:
		writer.writerow(['robot_data_path_euler', 'robot_data_path_quat', 'xela_1_data_path', 'xela_2_data_path', 'experiment_number', 'time_steps'])
	for row in path_file:
		writer.writerow(row)

