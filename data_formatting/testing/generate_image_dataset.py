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

data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/will_data_collection/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/test_dataset/'
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
	robot_task_space = []
	for state in robot_state[1:]:
		ee_orientation = R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)
		robot_task_space.append([state[-7], state[-6], state[-5], ee_orientation[0], ee_orientation[1], ee_orientation[2]])
	robot_task_space = np.asarray(robot_task_space).astype(float)

	# normalise for each value:
	min_max = []
	for feature in range(6):
		min_max.append([ min(np.array(robot_task_space[:,feature])), max(np.array(robot_task_space)[:,feature]) ])

	for time_step in range(robot_task_space.shape[0]):
		for feature in range(6):
			robot_task_space[time_step][feature] = (robot_task_space[time_step][feature] - min_max[feature][0]) / (min_max[feature][1] - min_max[feature][0])

	####################################### Xela Sensor Data ###########################################
	tactile_data = []
	for sample in xela_sensor1[1:]:
		tactile_data_sample_x, tactile_data_sample_y, tactile_data_sample_z = [], [], []
		for i in range(0, len(xela_sensor1[0]), 3):
			tactile_data_sample_x.append(float(sample[i]))
			tactile_data_sample_y.append(float(sample[i+1]))
			tactile_data_sample_z.append(float(sample[i+2]))
		tactile_data.append([tactile_data_sample_x, tactile_data_sample_y, tactile_data_sample_z])

	# mean starting values:
	tactile_mean_start_values = []
	tactile_offsets = []
	for feature in range(3):
		tactile_mean_start_values.append(int(sum(tactile_data[0][feature]) / len(tactile_data[0][feature])))
		tactile_offsets.append([tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data[0][feature]])

	# normalise for each force:
	tactile_data = np.array(tactile_data)
	min_max_tactile = []
	for feature in range(3):
		min_max_tactile.append([ min([min(x) for x in tactile_data[:,feature]]), max([max(x) for x in tactile_data[:,feature]]) ])

	for time_step in range(tactile_data.shape[0]):
		for feature in range(3):
			tactile_sample_test = [offset + real_value for offset, real_value in zip(tactile_offsets[feature], tactile_data[time_step][feature])]
			for i in range (tactile_data.shape[2]):
				tactile_data[time_step][feature][i] = (tactile_sample_test[i] - min_max_tactile[feature][0]) / (min_max_tactile[feature][1] - min_max_tactile[feature][0])

	# Create & save the images:
	tactile_images, tactile_image_names = [], []
	for time_step in range(tactile_data.shape[0]):
		# create the image:
		current_image = create_image(tactile_data[time_step][0], tactile_data[time_step][1], tactile_data[time_step][1])
		tactile_images.append(current_image)
		# save the image:
		image_name = "tactile_image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
		tactile_image_names.append(image_name)
		np.save(out_dir + image_name, current_image)

	####################################### Format data into time series ###########################################
	sequence_length = context_length + horrizon_length
	for sample in range(0, tactile_data.shape[0] - sequence_length):
		robot_data_euler_sequence, tactile_data_sequence, experiment_data_sequence, time_step_data_sequence, tactile_image_name_sequence = [], [], [], [], []
		for t in range(0, sequence_length):
			robot_data_euler_sequence.append(list(robot_task_space[sample+t]))
			tactile_data_sequence.append(list(tactile_data[sample+t]))
			tactile_image_name_sequence.append(tactile_image_names[sample + t])
			experiment_data_sequence.append(experiment_number)
			time_step_data_sequence.append(sample+t)

		####################################### Save the data and add to the map ###########################################
		np.save(out_dir + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
		np.save(out_dir + 'xela_1_data_' + str(index_to_save), tactile_data_sequence)
		np.save(out_dir + 'xela_1_image_data_' + str(index_to_save), tactile_image_name_sequence)
		np.save(out_dir + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
		np.save(out_dir + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
		ref = []
		ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
		ref.append('tactile_data_sequence' + str(index_to_save) + '.npy')
		ref.append('tactile_image_name_sequence' + str(index_to_save) + '.npy')
		ref.append('experiment_number_' + str(index_to_save) + '.npy')
		ref.append('time_step_data_' + str(index_to_save) + '.npy')
		path_file.append(ref)
		index_to_save += 1
	break

with open(out_dir + '/map.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'tactile_image_name_sequence', 'experiment_number', 'time_steps'])
	for row in path_file:
		writer.writerow(row)
