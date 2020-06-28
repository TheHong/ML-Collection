"""
This scripts loads, splits, and scales the maps dataset ready for training.
ONLY NEED TO RUN THIS SCRIPT ONCE to create the npz file.

Based on https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for...
-image-to-image-translation/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

from utils import get_path


def load_images(path, size=(256, 512)):
    """Images are 600x1200, with the left square being satellite and the right
    square being map. This function transforms it to a desired shape 
    (e.g. 256x512), and splits it.

    Args:
            path (str): Path to the npz file path
            size (tuple, optional): Size of . Defaults to (256,512).

    Returns:
            tuple(np.array): List of satellite images and map images
    """

    sat_list, map_list = list(), list()

    # enumerate filenames in directory, assume all are images
    for filename in os.listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)

        # convert to numpy array
        pixels = img_to_array(pixels)

        # split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        sat_list.append(sat_img)
        map_list.append(map_img)

    return np.asarray(sat_list), np.asarray(map_list)


def show_samples(filename, n_samples=3):
    """Shows samples of satellite and their corresponding map images

    Args:
            filename (str): Path to the npz file.
            n_samples (int, optional): Number of samples to display. Defaults to 3.
    """
    filename = get_path(filename)

    data = np.load(filename)
    sat_images, map_images = data['arr_0'], data['arr_1']
    print('Loaded: ', sat_images.shape, map_images.shape)

    # plot source images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(sat_images[i].astype('uint8'))

    # plot target image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(map_images[i].astype('uint8'))

    plt.show()


if __name__ == "__main__":
    filename = '../maps_256.npz'
    if not os.path.isfile(filename):  # npz file does not exist
        print("npz file not found. Will generate.")
        # Load images
        print("Loading...")
        path = get_path('../maps/train/')
        sat_images, map_images = load_images(path)
        print('Loaded: ', sat_images.shape, map_images.shape)

        # Save as compressed numpy array
        print("Saving...")
        np.savez_compressed(filename, sat_images, map_images)
        print('Saved dataset: ', filename)
    else:
        print("npz file found.")

    show_samples(filename)
