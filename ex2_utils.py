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
    # Sigma of kernel i,j = 1.0
    sigma = 1.0
    center = k_size // 2
    kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            ker_diff = np.sqrt(np.power(i - center, 2) + np.power(j - center, 2))
            kernel[i, j] = np.exp(-(np.power(ker_diff, 2)) / (2 * np.power(center, 2)))

    gaussian_kernel = kernel / sigma
    blur = conv2D(in_image, gaussian_kernel)
    return blur


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Creating a Gaussian kernel using the OpenCV library
    gaussian_kernel = cv2.getGaussianKernel(k_size, -1)
    # Applying the Gaussian kernel to the image
    blurred_img = cv2.sepFilter2D(in_image, -1, gaussian_kernel, gaussian_kernel)
    return blurred_img

# is this an edge - sends relevant pairs to check()
def check_area(list):
    answer = False
    for i in range(4):
        try:
           if check(list[i],list[8-i]):
               return True
           if list[4]<0 and list[i]>0:
               return True
        except:
            pass

    return answer

# checks if one value is positive and one is negative
def check(a, b):
    if (a>0 and b>0) or (a<0 and b<0) or (a==0 and b==0):
        return False
    return True

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    # create answer array
    answer = np.zeros(img.shape)

    # run gaussian blurring on image
    gaussian = cv2.GaussianBlur(img, (11,11), 0)

    # find derivative
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    img_f = conv2D(gaussian,laplacian)

    # zero crossing
    for i in range(img_f.shape[0]-1):
        for j in range(img_f.shape[1]-1):
            pixels = [img_f[i - 1][j - 1], img_f[i - 1][j], img_f[i - 1][j + 1],
                      img_f[i][j-1], img_f[i][j], img_f[i][j+1],
                      img_f[i+1][j-1], img_f[i+1][j], img_f[i+1][j+1]]
            if check_area(pixels):
                answer[i][j]=1

    return answer

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
    circles = list()
    thresh = 0.7
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, thresh)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh)
    direction = np.radians(np.arctan2(sobel_x, sobel_y) * 180 / np.pi)
    accumulator = np.zeros((len(img), len(img[0]), max_radius + 1))
    edges = cv2.Canny(np.uint8(img), 0.1, 0.45)
    height = len(edges)
    width = len(edges[0])
    for x in range(0, height):
        for y in range(0, width):
            if edges[x][y] == 255:
                for radius in range(min_radius, max_radius + 1):
                    angle = direction[x, y] - np.pi / 2
                    # x1, y1 => value + radius
                    # x2, y2 => value - radius
                    x1, x2 = np.int32(x - radius * np.cos(angle)), np.int32(x + radius * np.cos(angle))
                    y1, y2 = np.int32(y + radius * np.sin(angle)), np.int32(y - radius * np.sin(angle))
                    if 0 < x1 < len(accumulator) and 0 < y1 < len(accumulator[0]):
                        accumulator[x1, y1, radius] += 1
                    if 0 < x2 < len(accumulator) and 0 < y2 < len(accumulator[0]):
                        accumulator[x2, y2, radius] += 1

    thresh = np.multiply(np.max(accumulator), 1 / 2)
    x, y, radius = np.where(accumulator >= thresh)
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0 and radius[i] == 0:
            continue
        circles.append((y[i], x[i], radius[i]))

    return circles


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

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206664278
