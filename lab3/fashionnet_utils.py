import numpy as np
import scipy

def FFTConv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_c, input_h, input_w = inputs.shape
    output_c, _, kernel_h, kernel_w  = kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h + padding*2 - kernel_h) // stride + 1
    output_w = (input_w + padding*2 - kernel_w) // stride + 1

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((0,0), (padding,padding), (padding, padding)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_c, output_h, output_w))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[out_c,:,:] += scipy.signal.fftconvolve(inputs[in_c,:,:], kernels[out_c, in_c,:,:], mode=mode)[::stride, ::stride]
        output[out_c,:,:] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def Conv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
    input_c, input_h, input_w = inputs.shape
    print(kernels.shape)
    output_c, _, kernel_h, kernel_w  = kernels.shape
    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h + padding*2 - kernel_h) // stride + 1
    output_w = (input_w + padding*2 - kernel_w) // stride + 1

    # if padding
    if(padding > 0):
        inputs = np.pad(inputs, pad_width=((0,0), (padding,padding), (padding, padding)), mode='constant', constant_values=0)

    # get the output set up
    output = np.zeros(shape=(output_c, output_h, output_w))

    for out_c in range(output_c):
        for in_c in range(input_c):
            output[out_c,:,:] += scipy.signal.convolve2d(inputs[in_c,:,:], kernels[out_c, in_c,:,:], mode=mode)[::stride, ::stride]
        output[out_c,:,:] += biases[out_c]

    # apply relu
    if(activation == 'relu'):
        output = np.maximum(0, output)

    return output

def MaxPooling(inputs, kernel_size, stride):
    input_c, input_h, input_w = inputs.shape

    # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1

    output = np.zeros(shape=(input_c, output_h, output_w))

    # iterate through each item in output
    for out_h in range(output_h):
        for out_w in range(output_w):
            # get respective values in input
            h = out_h * stride
            w = out_h * stride
            # get 3d chunk of input
            inputs_chunk = inputs[:, h:h + kernel_size, w:w + kernel_size]
            output[:, out_h, out_w] = np.amax(inputs_chunk, axis=(1, 2))

    return output

def FullyConnected(inputs, weights, biases, activation='none'):
    if(activation == 'relu'):
        return np.maximum(0, np.dot(inputs, weights) + biases)
    else:
        return np.dot(inputs, weights) + biases

def Flatten(inputs):
    return inputs.flatten()