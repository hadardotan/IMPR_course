import numpy as np
from . import sol5_utils
# import sol5_utils
import scipy.ndimage.filters
from skimage import img_as_float
from skimage import color
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add
from tensorflow.keras.optimizers import Adam

from matplotlib.pyplot import imread
from skimage.color import rgb2gray

COLORS_FOR_PLOT = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'm'}
cache = {}



def read_image(filename):
    """
    :param filename: string containing the image filename to read
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image 1 or an RGB image 2
    :return: grayscale or RGB np.float64 image with values in range [0, 1]
    """

    im = imread(filename)
    if im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im)
    return im



def crop_images(images, crop_size):
    if images.ndim == 4:
        curr_images = images
    else:
        curr_images = images[np.newaxis]
    i = np.random.randint(curr_images.shape[1] - crop_size[0])
    j = np.random.randint(curr_images.shape[2] - crop_size[1])
    cropped = curr_images[:, i: int(i + crop_size[0]), j:int(j + crop_size[1])]
    if images.ndim != 4:
        cropped = np.squeeze(cropped, axis=0)
    return cropped



def corrupt_image(curr_image, corruption_func, crop_size):
    """

    :param curr_image:
    :param corruption_func:
    :param crop_size:
    :return:
    """
    corrupt_size = (3*crop_size[0], 3*crop_size[1])
    cropped = crop_images(curr_image, corrupt_size)
    corrupted = corruption_func(cropped)
    patches = crop_images(np.array([corrupted, cropped]), crop_size)
    return patches - 0.5


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: data_generator, a Python’s generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, height, width, 1)
    """
    while True:
        source_batch, target_batch = np.empty((batch_size,) + crop_size + (1,)), np.empty((batch_size,) + crop_size + (1,))
        for i in range(batch_size):
            idx = np.random.randint(len(filenames))
            filename = filenames[idx]
            if filename not in cache:
                cache[filename] = read_image(filename)
            curr_image = cache[filename][:, :, np.newaxis]
            source_patch, target_patch = corrupt_image(curr_image, corruption_func, crop_size)
            source_batch[i], target_batch[i] = source_patch, target_patch

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    takes as input a symbolic input tensor and the number of channels for each of its convolutional layers,
     and returns the symbolic output tensor of the layer configuration described above.
    :param input_tensor:
    :param num_channels:
    :return:
    """

    conv = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    relu = Activation('relu')(conv)
    conv2 = Conv2D(num_channels, (3, 3), padding='same')(relu)
    add = Add()([input_tensor, conv2])
    return add


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height:
    :param width:
    :param num_channels:
    :param num_res_blocks:
    :return: untrained Keras model , with input dimension the shape of (height, width, 1)4,
    and all convolutional layers with number of output channels equal to num_channels,
    except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    """

    if num_res_blocks == 0:
        return
    input = Input((height, width, 1))
    net = Conv2D(num_channels, (3, 3), padding='same')(input)
    net = Activation('relu')(net)
    residual = resblock(net, num_channels)  # number of output channels equal to num_channels
    # continue res as num of res block in nn
    for i in range(num_res_blocks - 1):
        residual = resblock(residual, num_channels)
    last = Conv2D(1, (3, 3), padding='same')(residual)
    last = Add()([input, last])
    model = Model(inputs=input, outputs=last)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    train the model
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files.
    :param corruption_func:
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    """

    # divide data to training and validation
    np_images = np.array(images)
    n = len(images)
    crop_size = model.input_shape[1:3]
    training_size = int(0.8*n)
    training = load_dataset(np_images[:training_size], batch_size, corruption_func, crop_size)
    validation = load_dataset(np_images[training_size:], batch_size, corruption_func, crop_size)

    # compile the model using the “mean squared error” loss and ADAM optimizer
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))

    # train the model using the generator
    model.fit_generator(training, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=validation,
                        validation_steps=(num_valid_samples/batch_size))


def restore_image(corrupted_image, base_model):
    """

    :param corrupted_image: a grayscale image of shape (height, width) and with values
    in the [0, 1] range of type float64
    :param base_model: a neural network trained to restore small patche
    :return:
    """

    # adjust the model to the new size
    new_corrupted = corrupted_image[:, :, np.newaxis] - 0.5
    input_image = Input(shape=new_corrupted.shape)
    output_image = base_model(input_image)
    adjusted_model = Model(inputs=input_image, outputs=output_image)

    # restore the image
    restored_image = np.array((adjusted_model.predict(new_corrupted[np.newaxis], batch_size=1) + 0.5)).astype(np.float64) # we subtracted 0.5 in the dataset generation.
    return restored_image.clip(0, 1)[0, :, :, 0]


"""
IMAGE DENOISING
"""

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    randomly sample a value of sigma, uniformly distributed between min_sigma and max_sigma, followed by adding to
    every pixel of the input image a zero-mean gaussian random variable with standard deviation equal to sigma
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma,
    representing the maximal variance of the gaussian distribution.
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    corrupted = image + np.random.normal(0, sigma, size=image.shape)
    corrupted = (1 / 255) * np.round(corrupted * 255)
    return corrupted.clip(0, 1)



def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    train a denoising model network
    :param num_res_blocks:
    :param quick_mode:
    :return:
    """

    # constants for denoising learning
    patch_size, num_channels, min_sigma, max_sigma = 24, 48, 0.0, 0.2
    if not quick_mode:
        batch_size, steps_per_epoch, num_epoch, num_valid_samples = 100, 100, 5, 1000
    else:
        batch_size, steps_per_epoch, num_epoch, num_valid_samples = 10, 3, 2, 30
    images = sol5_utils.images_for_denoising()

    # wrap the function add_gaussian_noise in a lambda expression to fit the protocol of corruption_func
    corruption_func = lambda image: add_gaussian_noise(image, min_sigma, max_sigma)

    # create and train denoising model
    denoising_model = build_nn_model(patch_size, patch_size, num_channels, num_res_blocks)
    train_model(denoising_model, images, corruption_func, batch_size, steps_per_epoch, num_epoch, num_valid_samples)

    return denoising_model


"""
IMAGE DEBLURRING
"""



def add_motion_blur(image, kernel_size, angle):
    """
    simulate motion blur on the given image using a square kernel of size kernel_size where the line
    has the given angle in radians, measured relative to the positive horizontal axis
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0, π)
    :return:
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    if image.ndim == 3 and kernel.ndim == 2:
        kernel = kernel[:, :, np.newaxis]
    corrupted = scipy.ndimage.filters.convolve(image, kernel)
    corrupted = (1 / 255) * np.round(255 * corrupted)
    return corrupted.clip(0, 1)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    random pick of params and then apply add_motion_blur
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return:
    """
    # sample an angle at uniform from the range [0, π)
    angle = np.random.uniform() * np.pi
    # chose a kernel size at uniform from the list list_of_kernel_sizes
    kernel_size = list_of_kernel_sizes[np.random.randint(len(list_of_kernel_sizes))]
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    train a denblurring model network
    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    patch_size, num_channels = 16, 32
    list_of_kernel_sizes = [7]
    if not quick_mode:
        batch_size, steps_per_epoch, num_epoch, num_valid_samples = 100, 100, 10, 1000
    else:
        batch_size, steps_per_epoch, num_epoch, num_valid_samples = 10, 3, 2, 30
    images = sol5_utils.images_for_deblurring()

    # wrap the function random_motion_blur in a lambda expression to fit the protocol of corruption_func
    corruption_func = lambda image: random_motion_blur(image, list_of_kernel_sizes)

    # create and train denblurring model
    deblurring_model = build_nn_model(patch_size, patch_size, num_channels, num_res_blocks)
    train_model(deblurring_model, images, corruption_func, batch_size, steps_per_epoch, num_epoch, num_valid_samples)

    return deblurring_model