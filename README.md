# cGAN
Implementation of a conditional generative adverserial network to translate between satellite image and Google maps image. This implementation was done as part of my undergraduate thesis in developing Sim2real Methods for Urban Search and Rescue Robots.

Based on https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/.

Paper is: https://arxiv.org/pdf/1611.07004.pdf

Data from: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz

## The Task
- Image sizes: 256x256
- Source: Satellite Image
- Target: Google Maps Image

- 1097 training images
- 1099 validation images

## About the cGAN

### Discriminator
Disriminator is a DNN that predicts likelihood of a target image is a real/fake translation of source (Essentially conditional image classification).

The discriminator in pix2pix is a __PatchGAN disciminator__: the design is based on "effective receptive field of the model, which defines the relationship between one output of the model to the number of pixels in the input image." It is designed so that in the model's output, each prediction element corresponds to a __70x70__ patch of the input image. In other words, each prediction element gives the likelihood that a patch of the input image is real. This results in the ability for the same model to be applied to input images of different sizes. As such, the __model output depends on the size of the input__.

### Training
Discriminator trained directly on real and generated images. 

Generator is trained via disriminator model. The loss for generator is weighted sum of two components:
1. Adversarial loss: Encourages the generation of images in target domain
2. L1 loss (mean absolute error between generated and source): Encourages the translation between source and target picture

#### Procedure
One epoch is one run through the batches. With batch size of 1 and 1097 training images, there will be 1097 iterations per epoch.
