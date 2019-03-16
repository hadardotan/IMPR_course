
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2grey




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



def imdisplay(filename, representation):
    """
    open a new figure and display the loaded image in the converted representation.
    :param filename: string containing the image filename to read
    :param representation:  representation code, either 1 or 2 defining whether the output should be a grayscale
    image 1 or an RGB image 2
    """
    im = read_image(filename, representation)
    if representation == 1:
        plt.imshow(im, cmap='gray')
        plt.show()
    if representation == 2:
        plt.imshow(im)
        plt.show()



def rgb2yiq(imRGB):
    """
    convert RGB image to YIQ image using transform matrix
    :param imRGB: RGB np.float64 image with shape  (height,width,3)
    :return: YIQ np.float64 image with shape  (height,width,3)
    """
    trans = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    return np.dot(imRGB, trans)


def yiq2rgb(imYIQ):
    """
    convert YIQ image to RGB image using transform matrix opposite
    :param imYIQ: YIQ np.float64 image with shape  (height,width,3)
    :return: RGB np.float64 image with shape  (height,width,3)
    """
    trans = np.array([[1, 0.956, 0.62], [1, -0.272, -0.647], [1, -1.108, 1.705]])
    return np.dot(imYIQ, trans)


def histogram_equalization_helper(im):
    """
    performs the equalization algorithm from class
    :param im:
    :return:
    """

    im *= (255 / im.max())
    c_m = im.min()
    hist_orig, bins = np.histogram(im, bins=256, range=[0, 256])
    cumulative_hist = np.cumsum(hist_orig)
    cumulative_hist = (((cumulative_hist - c_m) * 255) /(im.size)).astype(int)
    im_eq = cumulative_hist[im.astype(int)]
    hist_eq, bins_eq = np.histogram(im_eq, bins=256, range=[0, 256])
    im_eq = im_eq/ 255

    # plt.plot((bins[:-1] + bins[1:]) / 2, hist_orig)
    # plt.hist(im.flatten(), bins=128)
    # plt.show()
    #
    # plt.plot((bins_eq[:-1] + bins_eq[1:]) / 2, hist_eq)
    # plt.hist(im.flatten(), bins=128)
    #
    # plt.show()
    return im_eq, hist_orig, hist_eq


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image
    :param im_orig: grayscale or RGB float64 image with values in [0, 1]
    :return: im_eq: equalized image of im_orig
    :return: hist_orig: a 256 bin histogram of im_orig
    :return: hist_eq: a 256 bin histogram of im_eq
    """

    shape_len = len(im_orig.shape)
    if shape_len == 2:  # grayscale
        return histogram_equalization_helper(im_orig)
    elif shape_len == 3 and im_orig.shape[2] == 3:  # rgb
        im_yiq = rgb2yiq(im_orig)
        y = im_yiq[:, :, 0]
        y_eq, hist_orig, hist_eq = histogram_equalization_helper(y)
        im_yiq[:, :, 0] = y_eq
        im_eq = yiq2rgb(im_yiq)
        return im_eq, hist_orig, hist_eq

    else:
        print("error")
        return


def perform_quantization_loop(z, q, n_iter, hist, bins):
    """
    performs quantization loop until the process converges or at most n_iter rounds
    :param z: num of bins
    :param q: values to change values in bins
    :param n_iter: the maximum number of iterations of the optimization procedure
    :param hist: histogram to perform quantization process on
    :param bins: bins to perform quantization process on
    :return:
    """
    error = []
    current_error = 0
    for i in range(n_iter):
        sigma = np.zeros(len(q))
        for j in range(len(q)):
            p_z = hist[z[j]:z[j+1]+1]
            Z = bins[z[j]:z[j+1]+1]
            divided = sum(p_z*Z)
            divisor = sum(p_z)
            q[j] = divided / divisor
            Q = np.array([q[j]]*Z.size)
            sigma[j] = sum( (Q - Z) * (Q - Z) * p_z)

        if current_error == sum(sigma):  # process converged
            break

        else:
            current_error = sum(sigma)
            error.append(current_error)
            for j in range(1, len(z) - 1):
                z[j] = (q[j-1] + q[j]) / 2

    return z, q, error






def quantization_helper(im, n_quant, n_iter):
    """
    performs quantization procedure on grayscale image or 1 channel of rgb image
    :param im: float64 image with values in [0, 1].
    :param n_quant: number of intensities im_quant image would have.
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: im_quant: the quantized float64 image with values in [0, 1].
    :return: error: array with shape (n_iter,) (or less) of the total intensities error for each iteration of
     the quantization procedure.
    """
    im *= (255 / im.max())
    hist, bins = np.histogram(im, bins=256, range=[0, 256])
    cumulative_hist = np.cumsum(hist)
    # initial division such that each segment will contain approximately the same number of pixels.
    num_of_pixels = cumulative_hist.max() / n_quant
    z = np.zeros(shape=n_quant + 1, dtype='int')
    for i in range(0, len(z) - 1):
        z[i] = np.argmin(np.absolute(cumulative_hist - num_of_pixels * (i)))

    z[len(z) - 1] = 255 # The first and last elements are 0 and 255 respectively.
    q = np.zeros(shape=n_quant, dtype='float64')

    z, q, error = perform_quantization_loop(z, q, n_iter, hist, bins)
    lookup_table = np.array([0]*256,dtype='float64')

    for i in range(n_quant):
        lookup_table[z[i]:z[i+1]] = q[i]

    im_quant = lookup_table[im.astype(int)]
    return im_quant, error


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image
    :param im_orig: float64 image with values in [0, 1].
    :param n_quant: number of intensities im_quant image would have.
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: im_quant: the quantized float64 image with values in [0, 1].
    :return: error: array with shape (n_iter,) (or less) of the total intensities error for each iteration of
     the quantization procedure.
    """
    shape_len = len(im_orig.shape)
    if shape_len == 2:  # grayscale
        return quantization_helper(im_orig, n_quant, n_iter)

    elif shape_len == 3: # rgb
        im_yiq = rgb2yiq(im_orig)
        y = im_yiq[:, :, 0]
        y_quant, error = quantization_helper(y, n_quant, n_iter)
        y_quant = y_quant/ 255
        im_yiq[:, :, 0] = y_quant
        im_quants = yiq2rgb(im_yiq)
        return im_quants, error

def main():

    im = read_image("dog.jpg", 2)
    plt.imshow(im)
    plt.show()

    # im_eq, hist_orig, hist_eq = histogram_equalize(im)
    #
    # plt.imshow(im_eq)
    # plt.show()
    #

    im_quant, error = quantize(im, 3, 10)
    plt.imshow(im_quant)
    plt.show()








main()



