
import numpy as np

from helpers import summarize_performance, \
    load_real_samples, generate_real_samples, generate_fake_samples
from cgan import get_discriminator, get_generator, get_gan


# train pix2pix models
def train(discriminator, generator, gan, dataset, n_epochs=100, n_batch=1):
    """
    In paper, n_epochs=200 and n_batch=1.
    Dataset contain two np.arrays: the source images and the target images
    source images are always real
    target images can be real or fake
    """
    # Get length of the discriminator output square
    patch_len = discriminator.output_shape[1]

    # calculate the number of batches per training epoch
    batch_per_epoch = int(len(dataset[0]) / n_batch)

    # calculate the number of training iterations
    n_iter = batch_per_epoch * n_epochs

    # manually enumerate epochs
    for i in range(n_iter):
        # Select a batch of real samples
        [source, target_real], label_real = generate_real_samples(
            dataset, n_batch, patch_len)

        # Generate a batch of fake samples
        target_fake, label_fake = generate_fake_samples(
            generator, source, patch_len)

        # Update discriminator for real and generated samples
        d_loss1 = discriminator.train_on_batch(
            [source, target_real], label_real)  # Desire label_real
        d_loss2 = discriminator.train_on_batch(
            [source, target_fake], label_fake)  # Desire label_fake

        # Update the generator
        # Desire label_real and target_real
        g_loss, _, _ = gan.train_on_batch(source, [label_real, target_real])

        # summarize performance
        print(">{} ({:.1f}%), d1[{}] d2[{}] g[{}]".format(
            i + 1,
            (i + 1) / float(n_iter) * 100,
            np.round(d_loss1, 3),
            np.round(d_loss2, 3),
            np.round(g_loss, 3),
        ))

        if (i+1) % (n_iter // 10) == 0:
            summarize_performance(i, generator, dataset)


if __name__ == "__main__":
    # load image data
    dataset = load_real_samples('../maps_256.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)

    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]

    # define the models
    discriminator = get_discriminator(image_shape)
    generator = get_generator(image_shape)

    # define the composite model
    gan = get_gan(generator, discriminator, image_shape)

    # train model
    train(discriminator, generator, gan, dataset)
