import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:

    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    flipped = np.flip(in_signal)
    answer = []
    # add zeros at the beginning and end of vector
    for i in range(k_size.size):
        flipped.append(0)
        flipped.insert(flipped.size, 0)
    # multiply and sum vector and kernel
    # for every value in the vector
    for i in range(flipped.size-1):
        n = 0
    # for every value in the
        for j in range(k_size.size-1):
            n += k_size[j]*flipped[j+i]
        answer.append(n)

    return answer


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    flipped = np.flip(kernel)
    answer = []
    # add padding to image
    height = (flipped.shape[0]-1)//2
    width = (flipped.shape[1]-1)//2
    padded = np.pad(in_image,(width,width,height,height),'edge')

    # multiply parallel places and put in (x,y)
    new = (padded[2:5,5:6]*flipped).sum()
    # for every row
    for i in range(in_image.shape[0])-1:
    # for every column
        for j in range(in_image.shape[1])-1:
            answer[i][j] = (padded[i:i+flipped.shape[0],j:j+flipped.shape[1]]*flipped).sum()

    return answer


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    a = [1, 0,-1]
    b = np.transpose(a)
    deriv = conv2D(in_image, a)
    derivT = conv2D(in_image, b)

    mag = np.sqrt(np.power(np.linalg.matrix_power(deriv,2))+np.power(np.linalg.matrix_power(derivT,2)))
    dir = np.arctan(derivT/deriv)

    return dir, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
