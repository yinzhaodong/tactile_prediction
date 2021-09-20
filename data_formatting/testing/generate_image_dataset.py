# This script is used to convert the tactile dataset into a set of images.
# 	- It saves a map.csv file that points to all the correct images in the sequence of data.
#		-  Tactile sequence, d 1*48 -> 64*64*3 int8, [folder_name/image_name_0, folder_name/image_name_1, ..., folder_name/image_name_N]
#		-  Robot sequence, r 1*6 float16, [[x0,y0,z0,r0,p0,y0], [x1,y1,z1,r1,p1,y1], ..., [xN,yN,zN,rN,pN,yN]]
#		-  Slip Classification, s bool, [s0, s1, ..., sN]

import numpy
import cv2
