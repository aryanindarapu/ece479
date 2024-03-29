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

# Verify the model and load the weights into the net
print(model.summary())
print(len(model.layers))
model.load_weights("./code/weights/inception_keras_weights.h5")  # Has been translated from checkpoint

model.save('inception_resnet_model')

converter = tf.lite.TFLiteConverter.from_saved_model('inception_resnet_model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('../inception_resnet_model.tflite', 'wb') as f:
  f.write(tflite_model)
