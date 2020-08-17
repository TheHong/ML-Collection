"""
Based on https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for...
-image-to-image-translation/.
Paper is: https://arxiv.org/pdf/1611.07004.pdf
An alternate source: https://www.tensorflow.org/tutorials/generative/pix2pix
"""

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, \
    Concatenate, BatchNormalization, Dropout


# Discriminator ===============================================================
def get_discriminator(image_shape):
    """Creates the discriminator Keras model
    Args:
        image_shape (tuple[int]): 3-element shape of the discriminator input
    Returns:
        keras.models.Model: Compiled Keras model object
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)

    # Input
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()(  # concatenate images channel-wise
        [in_src_image, in_target_image]
    )

    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # Second last output layer
    d = Conv2D(512, (4, 4), padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # Model
    model = Model([in_src_image, in_target_image], patch_out)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0002, beta_1=0.5),
        loss_weights=[0.5]
    )
    return model


# Generator ===================================================================
def encoder_block(layer_in, n_filters, batchnorm=True):
    """Block containing a convolution layer, (possibly) batchnorm operation,
    and leaky relu activation. This is the building block of the encoder
    portion of the generator.
    Args:
        layer_in (keras.layers): Input to the encoder block
        n_filters (int): Number of convolution filters to use
            (i.e. depth of the output)
        batchnorm (bool, optional): Whether or not to apply
            batch normalization. Defaults to True.
    Returns:
        keras.layers: Output of the encoder block
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)

    # Downsampling
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(layer_in)
    # Conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)

    # Activation
    g = LeakyReLU(alpha=0.2)(g)

    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """Block containing a upconvolution layer, batchnorm operation,
    (possibly) dropout operation and relu activation. This is the building
    block of the decoder portion of the generator.
    Args:
        layer_in (keras.layers): Input to the decoder block
        skip_in (keras.layers): Skip connection from an encoder layer
        n_filters (int): Number of convolution filters to use
        dropout (bool, optional): Whether or not to apply dropout.
        Defaults to True.
    Returns:
        keras.layers: Output of the decoder block
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)

    # Upsampling
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same',
                        kernel_initializer=init)(layer_in)

    # Batch normalization
    g = BatchNormalization()(g, training=True)

    # Conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)

    # Merge with skip connection
    g = Concatenate()([g, skip_in])

    # Activation
    g = Activation('relu')(g)

    return g


def get_generator(image_shape=(256, 256, 3)):
    """Creates the discriminator Keras model. Follows the encoder-decoder model
    using U-Net architecture. Takes source image and generates target image
    Args:
        image_shape (tuple, optional): Shape of the input image.
            Defaults to (256,256,3).
    Returns:
        keras.models.Model: Compiled Keras model object
    """
    # Weight initialization
    init = RandomNormal(stddev=0.02)

    # Image input
    in_image = Input(shape=image_shape)

    # Encoder
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)

    # Bottleneck
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    # Decoder
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # Output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same',
                        kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)  # Values are between -1 and +1

    # Model
    # Loss is defined later in define_gan since generator is updated via the
    # composite model (i.e. GAN)
    model = Model(in_image, out_image)
    return model


# GAN =========================================================
def get_gan(generator, discriminator, image_shape):
    """Creates the composite model using the weights of existing standalone
    generator and discriminator. Here, generator gets the source image as
    input. Discriminator gets source image AND the output of the generator as
    input. The GAN is used to update the generator during training.
    Args:
        generator (keras.models.Model): Compiled Keras model object for
            the generator
        discriminator (keras.models.Model): Compiled Keras model object for
            the discriminator
        image_shape (tuple[int]): Shape of the image input
    Returns:
        keras.models.Model: Compiled Keras model object
    """
    # Set the discriminator weights to be untrainable
    discriminator.trainable = False

    # Define the source image
    source_img = Input(shape=image_shape)

    # Connect the source image to the generator input
    generator_output = generator(source_img)

    # Connect the source input and generator output to the discriminator input
    discriminator_output = discriminator([source_img, generator_output])

    # The loss is used to update the generator
    model = Model(source_img, [discriminator_output, generator_output])
    model.compile(
        loss=['binary_crossentropy', 'mae'],
        optimizer=Adam(lr=0.0002, beta_1=0.5),
        loss_weights=[1, 100]
    )
    return model


if __name__ == "__main__":
    # Looking at their architecture ===========================================
    d = get_discriminator((256, 256, 3))
    g = get_generator()

    d.summary()
    g.summary()
