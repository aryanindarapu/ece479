## for ECE479 ICC Lab2 Part3

'''
*Main Student Script*
'''

# Your works start here

# Import packages you need here
from inception_resnet import InceptionResNetV1Norm
import numpy as np
import tensorflow as tf

# Create a model
model = InceptionResNetV1Norm()
model.save('inception_resnet_model')

# Verify the model and load the weights into the net
print(model.summary())
print(len(model.layers))
import os
print(os.getcwd())
model.load_weights("./code/weights/inception_keras_weights.h5")  # Has been translated from checkpoint
