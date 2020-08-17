import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

from utils import get_path
from helpers import load_real_samples, load_one_image

# plot source, generated and target images


def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))

    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i])
        # show title
        plt.title(titles[i])
    plt.show()


def test_on_training_img(model_path):
    # load dataset
    all_source, all_target = load_real_samples('../maps_256.npz')
    print('Loaded', all_source.shape, all_target.shape)

    # load model
    model = load_model(get_path(model_path))

    # select random example
    ix = np.random.randint(0, len(all_source), 1)
    source_img, target_img = all_source[ix], all_target[ix]

    # generate image from source
    gen_image = model.predict(source_img)

    # plot all three images
    plot_images(source_img, gen_image, target_img)


def test_on_img(model_path, file_path):
    src_image = load_one_image('sample_satellite_img.jpg')
    print('Loaded', src_image.shape)

    # load model
    model = load_model(get_path(model_path))

    # generate image from source
    gen_image = model.predict(src_image)

    # scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0

    # plot the image
    plt.imshow(gen_image[0])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # test_on_training_img('../models/model_109600.h5')
    test_on_img('../models/model_109600.h5', 'sample_satellite_img.jpg')
