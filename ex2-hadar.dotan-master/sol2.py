import numpy as np
import scipy.signal
from scipy.misc import imread as imread, imsave
from skimage.color import rgb2grey
import math


"""
1 - Discrete Fourier Transform - DFT
"""


def DFT(signal):
    """
    :param signal: array of dtype float64 with shape (N,1)
    :return: complex Fourier signal
    """
    n = len(signal)
    u = np.arange(0, n).reshape(n, 1)
    x = np.arange(0, n).reshape(1, n)
    ux = np.dot(u, x)
    mat = np.e ** (-2 * ux * np.pi * complex(0, 1) / n)
    fourier_signal = np.dot(mat, signal)
    return fourier_signal


def IDFT(fourier_signal):
    """
    :param fourier_signal: array of dtype complex128 with shape (N,1)
    :return: complex signal
    """
    n = len(fourier_signal)
    u = np.arange(0, n).reshape(n, 1)
    x = np.arange(0, n).reshape(1, n)
    ux = np.dot(u, x)
    mat = np.e ** (2 * ux * np.pi * complex(0, 1) / n)
    mat /= n
    signal = np.dot(mat, fourier_signal)
    return np.real_if_close(signal)



def DFT2(image):
    """
    :param image: grayscale image of dtype float64
    :return: 2D array of dtype complex128
    """

    n, m = image.shape
    u = np.arange(0, n).reshape(n, 1)
    x = np.arange(0, n).reshape(1, n)
    ux = np.dot(u, x)
    mat_x = np.e ** (-2 * ux * np.pi * complex(0, 1) / n)

    v = np.arange(0, m).reshape(m, 1)
    y = np.arange(0, m).reshape(1, m)
    vy = np.dot(v, y)
    mat_y = np.e ** (-2 * vy * np.pi * complex(0, 1) / m)

    ex = np.dot(mat_x, image)
    ex = np.transpose(ex)
    exy = np.dot(mat_y, ex)
    exy = np.transpose(exy)

    return exy


def IDFT2(image):
    """
    :param fourier_image: 2D array of dtype complex128
    :return: grayscale image of dtype float64
    """

    n, m = image.shape

    u = np.arange(0, n).reshape(n, 1)
    x = np.arange(0, n).reshape(1, n)
    ux = np.dot(u, x)
    mat_x = np.e ** (2 * ux * np.pi * complex(0, 1) / n)
    mat_x /= n

    v = np.arange(0, m).reshape(m, 1)
    y = np.arange(0, m).reshape(1, m)
    vy = np.dot(v, y)
    mat_y = np.e ** (2 * vy * np.pi * complex(0, 1) / m)
    mat_y /= m

    ex = np.dot(mat_x, image)
    ex = np.transpose(ex)
    exy = np.dot(mat_y, ex)
    exy = np.transpose(exy)
    return exy


"""
2 - Image derivatives
"""


def conv_der(im):
    """
    :param im: grayscale images of type float64
    :return: magnitude : magnitude of the derivative, with the same dtype and shape
    """
    # derive the image in each direction separately using simple convolution with [1, 0, −1]
    horizontal = np.array([[1], [0], [-1]])
    dx = scipy.signal.convolve2d(im, horizontal, mode="same")

    vertical = horizontal.transpose()
    dy = scipy.signal.convolve2d(im, vertical, mode="same")

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    """
    :param im: float64 grayscale image
    :return: magnitude :magnitude of image derivatives using Fourier transform
    """

    # Compute the Fourier Transform F
    fourier_im = np.fft.fftshift(DFT2(np.array(im)))
    n, m = np.shape(fourier_im)

    # Multiply each Fourier coefficient F(u,v) by u

    # because highest frequency is n/2
    n_index = np.arange(math.floor(-(n / 2)), math.floor(n / 2), dtype='float64').reshape((n, 1))
    m_index = np.arange(math.floor(-(m / 2)), math.floor(m / 2), dtype='float64').reshape((m, 1))

    dx = IDFT2(np.fft.fftshift(2 * math.pi * complex(1, 1) * (1 / n) * (fourier_im * n_index)))
    dy = IDFT2(np.fft.fftshift(2 * math.pi * complex(1, 1) * (1 / m) * (fourier_im * m_index.T)))

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


"""
3 - Convolution theory
"""


def create_gaussian_kernel(kernel_size):
    """

    :param kernel_size:
    :return:
    """
    if kernel_size == 1:
        gaussian_kernel = np.array([[1]])
    else:
        vector = np.array([1, 1])
        basic_vector = vector.copy().T
        for i in range(kernel_size - 2):
            vector = scipy.signal.convolve(vector, basic_vector)
        vector = vector.reshape(kernel_size, 1)
        gaussian_kernel = np.dot(vector, vector.T)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    return gaussian_kernel

def blur_spatial(im, kernel_size):
    """
    performs image blurring using 2D convolution between the image f and a gaussian kernel g
    :param im: input image to be blurred (grayscale float64 image)
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (grayscale float64 image)
    """

    # create the 2D gaussian kernel g
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    im_blur = scipy.signal.convolve2d(im, gaussian_kernel, mode="same")
    return im_blur


def blur_fourier (im, kernel_size):
    """
    performs image blurring with gaussian kernel in Fourier space
    :param im: input image to be blurred (grayscale float64 image)
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer)
    :return: blurry image (grayscale float64 image)
    """

    n, m = np.shape(im)

    # create the 2D gaussian kernel g
    gaussian_kernel = create_gaussian_kernel(kernel_size)

    # pad g with zeros to bring it to the same shape as the image
    g = np.zeros((n, m))
    x_start = math.floor((n / 2) - (kernel_size / 2))
    y_start = math.floor((m / 2) - (kernel_size / 2))

    g[x_start:gaussian_kernel.shape[0] + x_start, y_start:gaussian_kernel.shape[1] + y_start] = gaussian_kernel
    # transform g to its Fourier representation G
    G = DFT2(g)

    # transform the image f to its Fourier representation F
    F = DFT2(im)

    # perform the inverse transform on the result
    im_blur = IDFT2(F*G)

    return np.real(np.fft.fftshift(im_blur))



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
        x = rgb2grey(new_image)
        return x
    if representation == 2:
        return new_image












