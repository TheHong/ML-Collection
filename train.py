
from helpers import summarize_performance
from load import load_real_samples
from model import generate_real_samples, generate_fake_samples, \
    define_discriminator, define_generator, define_gan

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

if __name__ == "__main__":
    # load image data
    dataset = load_real_samples('../maps_256.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)

    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]

    # define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)

    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)
    
    # train model
    train(d_model, g_model, gan_model, dataset)