import math

import cv as cv
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:

    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # flip image
    flipped = np.array([np.flip(in_signal)])
    # create answer array
    answer = np.array([])
    # add zeros at the beginning and end of vector
    for i in range(k_size.size-1):
        flipped = np.insert(flipped,0,0)
        flipped = np.append(flipped,[0])
    # multiply and sum vector and kernel
    # for every value in the vector
    for i in range(flipped.size-1):
        n = 0
    # for every value in the kernel
        for j in range(k_size.size):
            n += k_size[j]*flipped[j+i]
        answer = np.append(answer,[n])
    return answer


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # flip image
    flipped = np.flip(kernel)
    # create answer array
    answer = np.zeros((in_image.shape[0],in_image.shape[1]))

    # add padding to image
    height = (flipped.shape[0]-1)//2
    width = (flipped.shape[1]-1)//2
    padded = np.pad(in_image,((height,height),(width,width)),'edge')

    # for every row
    for i in range(in_image.shape[0]):
    # for every column
        for j in range(in_image.shape[1]):
    # multiply parallel places and put in (x,y)
            answer[i][j] = (padded[i:i+flipped.shape[0],j:j+flipped.shape[1]]*flipped).sum()

    return answer


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    a = np.array([[1, 0,-1]])
    b = np.transpose([a])
    deriv = conv2D(in_image, a)
    derivT = conv2D(in_image, b)

    dir = np.sqrt(np.power(deriv, 2) + np.power(derivT, 2))
    mag = np.arctan2(derivT,deriv)

    return (dir, mag)


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
    # upload image
    # Loads an image
    src = cv.imread(img)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        return -1

    # detect edges
    detect = cv.Canny(img,min_radius, max_radius)
    lines = cv.HoughLines(detect, 1, np.pi / 180, 150, None, 0, 0)
    cdst = cv.cvtColor(detect, cv.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    return cdst


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
