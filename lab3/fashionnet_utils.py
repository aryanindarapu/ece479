import numpy as np
import scipy

def FFTConv2D(inputs, kernels, biases, stride=1, padding=0, mode='valid', activation='none'):
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
            output[:,:, out_c] += scipy.signal.fftconvolve(inputs[:,:, in_c], kernels[:,:, in_c, out_c], mode=mode)[::stride, ::stride]
        #output[:,:, out_c] += biases[out_c]

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