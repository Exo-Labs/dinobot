import numpy as np
import sys
from PIL import Image 
import cv2
np.set_printoptions(threshold=sys.maxsize)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def image_processing(img):
    im = Image.open(r'%s' % img) 
    im = np.array(im)
    processed_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1 = 200, threshold2 = 200)
    return processed_image

def weights_generation():
    np.random.seed(1)
    weights = 2 * np.random.random((257,1)) - 1
    return weights

def training(img):
    inputs = image_processing(img)
    output = np.array([[1,1,1,0]]).T

    weights = weights_generation()
    for i in range(10000):
        input_layer = inputs
        output_layer = sigmoid(np.dot(input_layer, weights)).T

        error = output - output_layer
        adjustments = np.dot(input_layer.T, error.T * (output_layer * (1 - output_layer)).T)

        weights.resize(adjustments.shape)
        weights += adjustments

    print('Result after training:')
    print(output_layer)
    return weights

def work(img, weights):
    inputs = image_processing(img)
    output = np.array([[1,1,1,0]]).T

    output_layer = sigmoid(np.dot(inputs, weights)).T
    error = output - output_layer
    adjustments = np.dot(inputs.T, error.T * (output_layer * (1 - output_layer)).T)

    weights.resize(adjustments.shape)
    weights += adjustments
    print(output_layer)


training('1.png')
# work('2.png', weights)