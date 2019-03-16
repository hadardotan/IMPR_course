import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave
from skimage.color import rgb2gray
import os

"""
Helper functions
"""

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


def blur_spatial(im, kernel):
    """
    performs image blurring using 2D convolution between the image f and a gaussian kernel g
    :param im: input image to be blurred (grayscale float64 image)
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (grayscale float64 image)
    """
    curr = ndimage.filters.convolve(im, kernel)
    return ndimage.filters.convolve(curr, kernel.transpose())


def subsample(im):
    """
    :param im: 2d image
    :return: even rows and columns of im
    """
    return im[::2, ::2]


def expand_im(im, kernel):
    """
    :param im:
    :param kernel_size:
    :return:
    """
    # zero padding
    padded_im = np.zeros((im.shape[0]*2, im.shape[1]*2))
    padded_im[::2, ::2] = im
    # blur
    expand_im = blur_spatial(padded_im, kernel*2)
    return expand_im

"""
Image Pyramids
"""

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
        blur_im = blur_spatial(current_im, gaussian_kernel)
        current_im = subsample(blur_im)
        if min(np.shape(current_im)) < 16:
            break
        pyr.append(current_im)
        pyr_level += 1
    return pyr, gaussian_kernel


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter to be used in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element of the array is
     a grayscale image.
    :return: filter_vec : row vector of shape (1, filter_size) used for the pyramid construction
    """

    gaussian_pyr, gaussian_kernel = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    curr_im = im
    pyr_level = 0
    while pyr_level < (max_levels -1):
        if min(np.shape(curr_im)) < 16:
            break
        curr_im = gaussian_pyr[pyr_level] - expand_im(gaussian_pyr[pyr_level+1], gaussian_kernel)
        pyr.append(curr_im)
        pyr_level += 1
    # the last image on pyr is the same from gaussian pyr
    pyr.append(gaussian_pyr[pyr_level])
    return pyr, gaussian_kernel


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec:
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr
    :return: img : reconstructed image
    """
    coeff_lpyr = [lpyr[i] * coeff[i] for i in range(len(lpyr))]
    for i in range(len(coeff_lpyr)-2, -1, -1):  # loop reverse on pyramid to sum all hierarchies
        expand = expand_im(coeff_lpyr[i+1], filter_vec)
        coeff_lpyr[i] = coeff_lpyr[i] + expand
    return coeff_lpyr[i]  # last hierarch


def render_pyramid(pyr, levels):
    """
    :param pyr: a Gaussian or Laplacian pyramid as defined above
    :param levels: the number of levels to present in the result ≤ max_levels
    :return: res : a single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """

    # stretch pyramid to [0,1]

    for i in range(levels):
        im_min = np.min(pyr[i])
        pyr[i] -= im_min
        pyr[i] *= (1/np.max(pyr[i]))

    res = pyr[0]
    # stack each pyramid level horizontally after stretching the values to [0, 1]
    for i in range(1, levels):
        current_im = np.zeros((pyr[0].shape[0], pyr[i].shape[1]))  # height according to first level,
        #  width according to current level
        current_im[:pyr[i].shape[0], :] = pyr[i]
        res = np.hstack([res, current_im])

    return res


def display_pyramid(pyr, levels):
    """
    :param pyr: a Gaussian or Laplacian pyramid as defined above
    :param levels: the number of levels to present in the result ≤ max_levels
    """

    render_im = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(render_im, cmap='gray')


"""
Pyramid Blending
"""


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1: grayscale image to be blended
    :param im2: grayscale image to be blended
    :param mask: a boolean mask containing True and False representing which parts of im1 and im2 should appear
    in the resulting im_blend.
    :param max_levels: max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  size of the Gaussian filter which defining the filter used in the construction of the
    Laplacian pyramids of im1 and im2.
    :param filter_size_mask: size of the Gaussian filter which defining the filter used in the construction of the
    Gaussian pyramid of mask.
    :return: im_blend : blended image
    """

    # Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively

    lap_pyr_1, filter_vec_1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr_2, filter_vec_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    # Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64)
    gaussian_mask_pyr, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)

    # Construct the Laplacian pyramid Lout of the blended image for each level k by:

    lap_out_pyr = [0] * len(lap_pyr_1)

    for k in range(len(lap_pyr_1)):
        lap_out_pyr[k] = (gaussian_mask_pyr[k]*lap_pyr_1[k]) + ((1 - gaussian_mask_pyr[k])*lap_pyr_2[k])

    # Reconstruct the resulting blended image from the Laplacian pyramid Lout

    im_blend = laplacian_to_image(lap_out_pyr, filter_vec_mask, [1]*len(lap_pyr_1))
    return np.clip(im_blend, 0, 1)


def relpath(filename):
    """

    :param filename:
    :return:
    """
    return os.path.join(os.path.dirname(__file__), filename)


def color_images_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    do image blend on each rgb layer

    :param im1: grayscale image to be blended
    :param im2: grayscale image to be blended
    :param mask: a boolean mask containing True and False representing which parts of im1 and im2 should appear
    in the resulting im_blend.
    :param max_levels: max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  size of the Gaussian filter which defining the filter used in the construction of the
    Laplacian pyramids of im1 and im2.
    :param filter_size_mask: size of the Gaussian filter which defining the filter used in the construction of the
    Gaussian pyramid of mask.
    :return: im_blend : blended color image

    """

    blended_im = np.zeros(im1.shape)
    for i in range(3):
        blended_im[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size_im,
                                               filter_size_mask)

    return blended_im


def display_images(im1, im2, mask, blended):
    """
    display the two input images, the mask, and the resulting blended image in a single figure
    :param im1:
    :param im2:
    :param mask:
    :param blended:
    :return:
    """
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(im1)
    plt.subplot(222)
    plt.imshow(im2)
    plt.subplot(223)
    plt.imshow(mask, cmap='gray')
    plt.subplot(224)
    plt.imshow(blended)
    plt.show()


def blending_example1():
    """
    perform pyramid blending on two sets of image pairs and masks
    display (using plt.imshow()) the two input images, the mask, and the resulting blended image in a single figure
    :return: im1
    :return: im2
    :return: mask
    :return: im_blend
    """
    ner = read_image(relpath("ner.jpg"), 2)
    pig = read_image(relpath("new_pig.jpg"), 2)
    mask = read_image(relpath("mask.jpg"), 1)
    mask = mask >= 0.5
    blended_pig_ner = color_images_blending(ner, pig, mask, 5, 15, 15)
    mask = mask.astype(bool)
    display_images(ner, pig, mask, blended_pig_ner)
    return ner, pig, mask, blended_pig_ner


def blending_example2():
    """
    perform pyramid blending on two sets of image pairs and masks
    display (using plt.imshow()) the two input images, the mask, and the resulting blended image in a single figure
    :return: im1
    :return: im2
    :return: mask
    :return: im_blend
    """
    liberty = read_image(relpath("liberty.jpg"), 2)
    icecream = read_image(relpath("icecream.jpg"), 2)
    mask = read_image(relpath("ice_mask.jpg"), 1)
    mask = mask >= 0.5
    blended_liberty_icecream = color_images_blending(liberty, icecream, mask, 3, 3, 3)
    display_images(liberty, icecream, mask, blended_liberty_icecream)
    return liberty, icecream, mask.astype(bool), blended_liberty_icecream



