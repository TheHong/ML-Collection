# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from matplotlib import pyplot

# load all images in a directory into memory
def load_images(path, size=(256,512)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [asarray(src_list), asarray(tar_list)]


# load and prepare training images
def load_real_samples(filename):
    """
    return a list of two NumPy arrays: 
    the first for source images and the second for corresponding target images.
    """
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    """    
    Select a batch of random samples, returns images and target.
    prepare a batch of random pairs of images from the training dataset, 
    and the corresponding discriminator label of class=1 to indicate they are real.

    """
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


if __name__ == "__main__":
    # # dataset path
    # path = '../maps/train/'
    # # load dataset
    # [src_images, tar_images] = load_images(path)
    # print('Loaded: ', src_images.shape, tar_images.shape)
    # # save as compressed numpy array
    # filename = 'maps_256.npz'
    # savez_compressed(filename, src_images, tar_images)
    # print('Saved dataset: ', filename)

    # load the dataset
    try:
        data = load('../maps_256.npz')
    except FileNotFoundError:
        print("Make sure this script is being run in the same folder as this script.")
    else:
        src_images, tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', src_images.shape, tar_images.shape)
        # plot source images
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(src_images[i].astype('uint8'))
        # plot target image
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(tar_images[i].astype('uint8'))
        pyplot.show()
