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

# Hyperparameters:
train_data_dir = '/home/user/Robotics/Data_sets/data_collection_preliminary/train/'
test_data_dir  = '/home/user/Robotics/Data_sets/data_collection_preliminary/test/'
train_out_dir  = '/home/user/Robotics/Data_sets/data_collection_preliminary/train_image_dataset_10c_10h/'
test_out_dir   = '/home/user/Robotics/Data_sets/data_collection_preliminary/test_image_dataset_10c_10h/'
scaler_out_dir = '/home/user/Robotics/Data_sets/data_collection_preliminary/scalar_info/'

context_length = 10
horrizon_length = 10
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

def calculate_scale_parameters():
	tactile_data_final = []
	robot_task_space_final = []
	for experiment_number in tqdm(range(len(files))):
		robot_task_space = []
		tactile_data = []
		robot_state = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
		xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))

		####################################### Robot Data ###########################################
		for state in robot_state[1:]:
			ee_orientation = R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)
			robot_task_space.append(
				[state[-7], state[-6], state[-5], ee_orientation[0], ee_orientation[1], ee_orientation[2]])
		robot_task_space = np.asarray(robot_task_space).astype(float)

		# normalise for each value:
		min_max = []
		for feature in range(6):
			min_max.append([ min(np.array(robot_task_space[:,feature])), max(np.array(robot_task_space)[:,feature]) ])
		for time_step in range(robot_task_space.shape[0]):
			for feature in range(6):
				robot_task_space[time_step][feature] =(robot_task_space[time_step][feature] - min_max[feature][0]) /(min_max[feature][1] - min_max[feature][0])

		####################################### Xela Sensor Data ###########################################
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
				for i in range(tactile_data.shape[2]):
					tactile_data[time_step][feature][i] =(tactile_sample_test[i] - min_max_tactile[feature][0]) /(min_max_tactile[feature][1] - min_max_tactile[feature][0])

		tactile_data_final += list(tactile_data)
		robot_task_space_final += list(robot_task_space)

	####################################### Find the scalar values ###########################################
	tactile_data_final = np.array(tactile_data_final)
	robot_task_space_final = np.array(robot_task_space_final)

	# tactile scalars: standard then min max:
	tactile_standard_scaler = [preprocessing.StandardScaler().fit(tactile_data_final[:,0]),
							   preprocessing.StandardScaler().fit(tactile_data_final[:,1]),
							   preprocessing.StandardScaler().fit(tactile_data_final[:,2])]
	tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tactile_standard_scaler[0].transform(tactile_data_final[:,0])),
							  preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tactile_standard_scaler[1].transform(tactile_data_final[:,1])),
							  preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tactile_standard_scaler[2].transform(tactile_data_final[:,2]))]

	# robot task space scalars: standard then min max:
	robot_min_max_scalar = []
	for feature in range(robot_task_space_final.shape[1]):
		robot_min_max_scalar.append(preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(robot_task_space_final[:,feature].reshape(-1, 1)))

	return tactile_standard_scaler, tactile_min_max_scalar, robot_min_max_scalar

if __name__  == "__main__":
	# read all the folder names for each trial
	files = glob.glob(train_data_dir + '/*')
	files += glob.glob(test_data_dir + '/*')
	index_to_save = 0

	# calculate the scalar parameters:
	tactile_standard_scaler, tactile_min_max_scalar, robot_min_max_scalar = calculate_scale_parameters()

	# save the scalars
	dump(tactile_standard_scaler[0], open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', 'wb'))
	dump(tactile_standard_scaler[1], open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', 'wb'))
	dump(tactile_standard_scaler[2], open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', 'wb'))
	dump(tactile_min_max_scalar[0], open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', 'wb'))
	dump(tactile_min_max_scalar[1], open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', 'wb'))
	dump(tactile_min_max_scalar[2], open(scaler_out_dir + 'tactile_min_max_scalar.pkl', 'wb'))

	dump(robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'))
	dump(robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'))
	dump(robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'))
	dump(robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'))
	dump(robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'))
	dump(robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'))

	# Do twice, for test & train:
	for path, data_dir_ in zip([train_out_dir, test_out_dir], [train_data_dir, test_data_dir]):
		files = glob.glob(data_dir_ + '/*')
		path_file = []

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
					robot_task_space[time_step][feature] =(robot_task_space[time_step][feature] - min_max[feature][0]) /(min_max[feature][1] - min_max[feature][0])

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
					for i in range(tactile_data.shape[2]):
						tactile_data[time_step][feature][i] =(tactile_sample_test[i] - min_max_tactile[feature][0]) /(min_max_tactile[feature][1] - min_max_tactile[feature][0])

			####################################### Scale the data ###########################################
			for index,(standard_scaler, min_max_scalar) in enumerate(zip(tactile_standard_scaler, tactile_min_max_scalar)):
				tactile_data[:, index] = standard_scaler.transform(tactile_data[:, index])
				tactile_data[:, index] = min_max_scalar.transform(tactile_data[:, index])

			for index, min_max_scalar in enumerate(robot_min_max_scalar):
				robot_task_space[:, index] = np.squeeze(min_max_scalar.transform(robot_task_space[:, index].reshape(-1, 1)))

			# Create & save the images:
			tactile_images, tactile_image_names = [], []
			for time_step in range(tactile_data.shape[0]):
				# create the image:
				current_image = create_image(tactile_data[time_step][0], tactile_data[time_step][1], tactile_data[time_step][1])
				tactile_images.append(current_image)
				# save the image:
				image_name = "tactile_image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
				tactile_image_names.append(image_name)
				np.save(path + image_name, current_image)

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
				np.save(path + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
				np.save(path + 'tactile_data_sequence_' + str(index_to_save), tactile_data_sequence)
				np.save(path + 'tactile_image_name_sequence_' + str(index_to_save), tactile_image_name_sequence)
				np.save(path + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
				np.save(path + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
				ref = []
				ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
				ref.append('tactile_data_sequence_' + str(index_to_save) + '.npy')
				ref.append('tactile_image_name_sequence_' + str(index_to_save) + '.npy')
				ref.append('experiment_number_' + str(index_to_save) + '.npy')
				ref.append('time_step_data_' + str(index_to_save) + '.npy')
				path_file.append(ref)
				index_to_save += 1

		with open(path + '/map.csv', 'w') as csvfile:
			writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
			writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'tactile_image_name_sequence', 'experiment_number', 'time_steps'])
			for row in path_file:
				writer.writerow(row)