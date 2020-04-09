import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

from utils import get_path


def load_one_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(get_path(filename), target_size=size)

	# convert to numpy array
	pixels = img_to_array(pixels)

	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5

	# reshape to 1 sample
	pixels = np.expand_dims(pixels, 0)

	return pixels


def load_real_samples(filename, is_sat2map=True):
	"""
	Prepares training images: source images and their corresponding target images.
	"""

	# load and unpack compressed arrays
	data = np.load(get_path(filename))
	satellite_imgs, map_imgs = data['arr_0'], data['arr_1']

	# scale from [0,255] to [-1,1]
	all_satellite = (satellite_imgs - 127.5) / 127.5
	all_map = (map_imgs - 127.5) / 127.5

	if is_sat2map:
		all_source = all_satellite
		all_target = all_map
	else:
		all_source = all_map
		all_target = all_satellite

	return [all_source, all_target]


def generate_real_samples(dataset, n_samples, patch_len):
	"""	
	Real samples come from training set.
	Prepare a batch of random pairs of images from the training dataset and their labels.
	For the discriminator
	"""
	
	# unpack dataset
	all_source, all_target = dataset

	# Randomly select images
	ix = np.random.randint(0, all_source.shape[0], n_samples)
	source, target = all_source[ix], all_target[ix]

	# Labels denoting "real"
	y = np.ones((n_samples, patch_len, patch_len, 1))

	return [source, target], y


def generate_fake_samples(generator, samples, patch_len):
	"""
	Fake samples come from generator
	"""

	# Generate fake instance
	X = generator.predict(samples)

	# Labels denoting "fake" 
	y = np.zeros((len(X), patch_len, patch_len, 1))

	return X, y


def summarize_performance(step, generator, dataset, n_samples=3):
	"""
	Generate samples and save as a plot and save the model
	"""

	if not os.path.isdir("results"):
		os.makedirs("results")
	if not os.path.isdir("../models"):
		os.makedirs("../models")
	plot_name = 'results/plot_%06d.png' % (step+1)
	model_name = '../models/model_%06d.h5' % (step+1)

	# select a sample of input images and fake samples
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	X_fakeB, _ = generate_fake_samples(generator, X_realA, 1)

	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])

	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])

	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])

	# save plot to file and save generator
	plt.savefig(plot_name)
	generator.save(model_name)

	plt.close()
	print('>Saved: %s and %s' % (plot_name, model_name))
