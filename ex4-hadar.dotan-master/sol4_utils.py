import numpy as np
from scipy import signal, ndimage
from scipy.misc import imread as imread
from skimage.color import rgb2gray


def read_image(filename, representation):
    """
    :param filename: string containing the image filename to read
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image 1 or an RGB image 2
    :return: grayscale or RGB np.float64 image with values in range [0, 1]
    """
    image = imread(filename)
    new_image = image.astype(np.float64)
    new_image /= 255
    if representation == 1:
        x = rgb2gray(new_image)
        return x
    if representation == 2:
        return new_image


def create_gaussian_kernel(kernel_size):
    """
    :param kernel_size:
    :return:
    """

    if kernel_size == 1:
        return np.array([[1]])
    vec = np.array([[1, 1]]).astype(np.float64)
    vec_to_return = vec.copy()
    for i in range(kernel_size - 2):
            vec_to_return = signal.convolve(vec_to_return, vec)
    normal = 1 / np.sum(vec_to_return)
    return normal * vec_to_return

def subsample(im):
    """
    :param im: 2d image
    :return: even rows and columns of im
    """
    return im[::2, ::2]


def blur_spatial(im, kernel):
    """
    performs image blurring using 2D convolution between the image f and a gaussian kernel g
    :param im: input image to be blurred (grayscale float64 image)
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (grayscale float64 image)
    """
    gaussian_kernel = create_gaussian_kernel(kernel)
    return signal.convolve2d(im, gaussian_kernel, mode='same')


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter to be used in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element of the array is
     a grayscale image.
    :return: filter_vec : row vector of shape (1, filter_size) used for the pyramid construction
    """
    pyr = [im]
    gaussian_kernel = create_gaussian_kernel(filter_size)
    current_im = im
    pyr_level = 1
    while pyr_level < max_levels:
        blur_im = blur_spatial(current_im, filter_size)
        current_im = subsample(blur_im)
        if min(np.shape(current_im)) < 16:
            break
        pyr.append(current_im)
        pyr_level += 1
    return pyr, gaussian_kernel

