"""



"""

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, \
	Concatenate, BatchNormalization, Dropout


# Discriminator ===========================
def define_discriminator(image_shape):
	"""
	70×70 PatchGAN discriminator.

	Takes two input images that are concatenated together
	Outputs a patch of predictions.

	"The model is optimized using binary cross entropy, and a weighting is used so that updates 
	to the model have half (0.5) the usual effect. 
	The authors of Pix2Pix recommend this weighting of model updates to slow down changes to the 
	discriminator, relative to the generator model during training."
	"""

	# weight initialization
	init = RandomNormal(stddev=0.02)

	# Input
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	merged = Concatenate()([in_src_image, in_target_image]) # concatenate images channel-wise

	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)

	# Model
	model = Model([in_src_image, in_target_image], patch_out)
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	
	return model


# Generator ===========================
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)

	# Downsampling
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

	# Conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)

	# Activation
	g = LeakyReLU(alpha=0.2)(g)

	return g


def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)

	# Upsampling
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	
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


def define_generator(image_shape=(256,256,3)):
	"""
	encoder-decoder model using U-Net architecture.

	Takes source image (satellite photo)
	Generates target image (Google maps image)
	"""

	# weight initialization
	init = RandomNormal(stddev=0.02)

	# image input
	in_image = Input(shape=image_shape)

	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)

	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)

	# decoder model
	d1 = define_decoder_block(b, e7, 512)
	d2 = define_decoder_block(d1, e6, 512)
	d3 = define_decoder_block(d2, e5, 512)
	d4 = define_decoder_block(d3, e4, 512, dropout=False)
	d5 = define_decoder_block(d4, e3, 256, dropout=False)
	d6 = define_decoder_block(d5, e2, 128, dropout=False)
	d7 = define_decoder_block(d6, e1, 64, dropout=False)

	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g) # Result is that values are between -1 and +1

	# Define model. Loss is defined later in define_gan since generator is updated via the composite model (i.e. GAN)
	model = Model(in_image, out_image)

	return model


# GAN =========================================================
def define_gan(generator, discriminator, image_shape):
	"""
	Creating the composite model using the weights of existing standalone generator and discriminator.
	Here, generator gets the source image. Discriminator gets source image AND the output of the generator.
	This is used to update the generator.
	"""

	# Set the discriminator weights to be untrainable
	discriminator.trainable = False

	# define the source image
	source_img = Input(shape=image_shape)

	# connect the source image to the generator input
	generator_output = generator(source_img)

	# connect the source input and generator output to the discriminator input
	discriminator_output = discriminator([source_img, generator_output])

	# The loss is used to uupdate the generator
	model = Model(source_img, [discriminator_output, generator_output])
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1, 100])
	
	return model