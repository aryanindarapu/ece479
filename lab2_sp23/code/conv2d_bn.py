# -*- coding: utf-8 -*-
## for ECE479 ICC Lab2 Part3

'''
*Definition for 2 dimension convolution layer with batch normalization*
'''

from modules import *


def conv2d_bn(x,
              filters, # number of output channels
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):

    ## DO NOT TOUCH
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    ##############################################################

    ## TO DO Step1 : Apply a Conv 2D Keras Layer with all given parameters
    # Your code goes here





    if not use_bias:

        bn_name = generate_layer_name('BatchNorm', prefix=name)
        ## TO DO Step 2 : Apply a Batch Normalization Keras Layer with all given parameters
        # Your code goes here




    if activation is not None:
        ac_name = generate_layer_name('Activation', prefix=name)
        ## TO DO Step 3 : Apply an Activation Keras Layer with all given parameters
        # Your code goes here




    ###############################################################
    return x