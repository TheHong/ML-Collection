"""
This scripts loads, splits, and scales the maps dataset ready for training.
Only need to do once to create the npz file.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

from utils import get_path

def load_images(path, size=(256,512)):
	"""
	Images are 600x1200, with the left square being satellite and the right square being map.
	This function transforms it to 256x512, and splits it.
	"""

	src_list, tar_list = list(), list()

	# enumerate filenames in directory, assume all are images
	for filename in os.listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)

		# convert to numpy array
		pixels = img_to_array(pixels)

		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)

	return np.asarray(src_list), np.asarray(tar_list)


def show_samples(filename, n_samples=3):
	filename = get_path(filename)

	data = np.load(filename)
	src_images, tar_images = data['arr_0'], data['arr_1']
	print('Loaded: ', src_images.shape, tar_images.shape)

	# plot source images
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(src_images[i].astype('uint8'))

	# plot target image
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(tar_images[i].astype('uint8'))

	plt.show()

if __name__ == "__main__":
	filename = '../maps_256.npz'
	if not os.path.isfile(filename): # npz file does not exist
		print("npz file not found. Will generate.")
		# Load images
		print("Loading...")
		path = get_path('../maps/train/')
		src_images, tar_images = load_images(path)
		print('Loaded: ', src_images.shape, tar_images.shape)

		# Save as compressed numpy array
		print("Saving...")
		np.savez_compressed(filename, src_images, tar_images)
		print('Saved dataset: ', filename)
	else:
		print("npz file found.")

	show_samples(filename)