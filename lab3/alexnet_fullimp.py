# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy
import numpy as np
import cv2
import time

# create conv2d_time, conv2d_fft, maxpool, and dense layers

# first convolutional layer: input of 3x227x227 convolved with 96 3x11x11 kernels with stride 4
# inputs is height, width, channels
# kernels is height, width, input channels, output channels (translates to: for each output channels there are weights for height, width, and channel)
def FFTConv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_h, input_w, input_c = inputs.shape
    kernel_h, kernel_w, _, output_c= kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h + padding*2 - kernel_h) // stride + 1
    output_w = (input_w + padding*2 - kernel_w) // stride + 1

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((padding,padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_h, output_w, output_c))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[:,:, out_c] += scipy.signal.fftconvolve(inputs[:,:, in_c], kernels[:,:, in_c, out_c], mode=mode)[::stride, ::stride]
        output[:,:, out_c] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def Conv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_h, input_w, input_c = inputs.shape
    kernel_h, kernel_w, _, output_c= kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    if(mode != 'same'):
        output_h = (input_h + padding*2 - kernel_h) // stride + 1
        output_w = (input_w + padding*2 - kernel_w) // stride + 1
    else:
        output_h = input_h // stride
        output_w = input_w // stride

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((padding,padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_h, output_w, output_c))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[:,:, out_c] += scipy.signal.convolve(inputs[:,:, in_c], kernels[:,:, in_c, out_c], mode=mode, method='direct')[::stride, ::stride]
        output[:,:, out_c] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def MaxPooling(inputs, kernel_size, stride):
    input_h, input_w, input_c = inputs.shape

    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1

    output = np.zeros(shape=(output_h, output_w, input_c))

    # iterate through each item in output
    for out_h in range(output_h):
        for out_w in range(output_w):
            # get respective values in input
            h = out_h * stride
            w = out_h * stride
            # get 3d chunk of input
            inputs_chunk = inputs[h:h + kernel_size, w:w + kernel_size, :]
            output[out_h, out_w, :] = np.amax(inputs_chunk, axis=(0, 1))

    return output

def FullyConnected(inputs, weights, biases, activation='none'):
    if(activation == 'relu'):
        return np.maximum(0, inputs @ weights + biases)
    elif(activation=='softmax'):
        return scipy.special.softmax(inputs @ weights + biases)
    else:
        return inputs @ weights + biases

def Flatten(inputs):
    return inputs.flatten()

inputs = np.zeros(shape=(3,227,227))

# load the weights and biases for all of the layers
conv1_b = np.load('./alexnet_model/conv1_b.npy')
conv1_w = np.load('./alexnet_model/conv1_w.npy')
conv2_b = np.load('./alexnet_model/conv2_b.npy')
conv2_w = np.load('./alexnet_model/conv2_w.npy')
conv3_b = np.load('./alexnet_model/conv3_b.npy')
conv3_w = np.load('./alexnet_model/conv3_w.npy')
conv4_b = np.load('./alexnet_model/conv4_b.npy')
conv4_w = np.load('./alexnet_model/conv4_w.npy')
conv5_b = np.load('./alexnet_model/conv5_b.npy')
conv5_w = np.load('./alexnet_model/conv5_w.npy')
fc6_b = np.load('./alexnet_model/fc6_b.npy')
fc6_w = np.load('./alexnet_model/fc6_w.npy')
fc7_b = np.load('./alexnet_model/fc7_b.npy')
fc7_w = np.load('./alexnet_model/fc7_w.npy')
fc8_b = np.load('./alexnet_model/fc8_b.npy')
fc8_w = np.load('./alexnet_model/fc8_w.npy')

# read the labels from a text file
with open('./labels.txt') as f:
    labels = f.readlines()

labels = [x.replace('"', '') for x in labels]
labels = [x.replace('\n', '') for x in labels]

# read an image to run inference on
im = cv2.imread('dog2.png')
im = im.reshape((3,227,227))


# start = time.perf_counter()
# outFFT = FFTConv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
# print(f"FFTConv2D took {time.perf_counter()-start} seconds")

# start = time.perf_counter()
# outDirect = Conv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
# print(f"Conv2D took {time.perf_counter()-start} seconds")

# avDiff = np.average(outFFT - outDirect)
# avWeights = np.average(conv1_w)
# minWeight = np.min(conv1_w)

# print(f"The average difference in output tensor is {avDiff}. This is {avDiff/avWeights}% of the average of all weights and {avDiff/minWeight}% of the smallest weight.")


# building alexnet
# out = FFTConv2D(im, conv1_w, conv1_b, stride=4, activation='relu')
# out = MaxPooling(out, 3, 2)
# out = Conv2D(out, conv2_w, conv2_b, stride=1, padding=2, activation='relu')
# out = MaxPooling(out, 3, 2)
# out = Conv2D(out, conv3_w, conv3_b, stride=1, padding=1, activation='relu')
# out = Conv2D(out, conv4_w, conv4_b, stride=1, padding=1, activation='relu')
# out = Conv2D(out, conv5_w, conv5_b, stride=1, padding=1, activation='relu')
# out = MaxPooling(out, 3, 2)
# out = Flatten(out)
# out = FullyConnected(out, fc6_w, fc6_b, activation='relu')
# out = FullyConnected(out, fc7_w, fc7_b, activation='relu')
# out = FullyConnected(out, fc8_w, fc8_b, activation='relu')

# guess = np.argmax(out)
# print(labels[guess])
fashion_mnist = keras.datasets.fashion_mnist
(train_val_images, train_val_labels), (test_images, test_labels) = fashion_mnist.load_data()

#preprocess the data
split = 50000
#split into validation and normal training
validation_images = train_val_images[split:]
train_images = train_val_images[:split]

validation_labels = train_val_labels[split:]
train_labels = train_val_labels[:split]

train_images = train_images / 255.0
validation_images = validation_images/255.0
test_images = test_images / 255.0

# reshape data
train_images = train_images[..., np.newaxis]
validation_images = validation_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fnconv1_w = np.load("fnconv1_w.npy")
fnconv2_w = np.load("fnconv2_w.npy")
fnconv1_b = np.load("fnconv1_b.npy")
fnconv2_b = np.load("fnconv2_b.npy")
fc1_w = np.load("fc1_w.npy")
fc1_b = np.load("fc1_b.npy")
fc2_w = np.load("fc2_w.npy")
fc2_b = np.load("fc2_b.npy")
fc3_w = np.load("fc3_w.npy")
fc3_b = np.load("fc3_b.npy")

count = 0
for i in range(100):
    first_im = test_images[i]

    # building fashionnet
    out = Conv2D(first_im, fnconv1_w, fnconv1_b, stride=1, activation='relu', mode='same')
    out = MaxPooling(out, 2, 2)
    out = Conv2D(out, fnconv2_w, fnconv2_b, stride=1, activation='relu', mode='valid')
    out = MaxPooling(out, 2, 2)
    out = Flatten(out)
    out = FullyConnected(out, fc1_w, fc1_b, activation='relu')
    out = FullyConnected(out, fc2_w, fc2_b, activation='relu')
    out = FullyConnected(out, fc3_w, fc3_b, activation='softmax')
    
    guess = np.argmax(out)

    if(guess == test_labels[i]):
        count += 1

print(count/100)
