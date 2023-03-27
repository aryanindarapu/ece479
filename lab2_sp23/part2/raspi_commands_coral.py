import tflite_runtime.interpreter as tflite
import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import time

test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# data preprocessing to normalize data
# test_images = test_images / 255.0

# reshape data
test_images = test_images[..., np.newaxis]
test_images = test_images.astype(np.uint8)

# Load the TFLite model and allocate tensors.
interpreter = edgetpu.make_interpreter("my_model_full_int_quant.tflite")
interpreter.allocate_tensors()

num_correct = 0
total_time = 0
for i in range(len(test_images)):
    # Test the model on random input data.
    common.set_input(interpreter, test_images[i].reshape((1, 28, 28, 1)))
    
    # Run the model
    start_time = time.time()
    interpreter.invoke()
    total_time += time.time() - start_time

    classes = classify.get_classes(interpreter, top_k=1)
    if classes[0].id == test_labels[i]: num_correct += 1

print(f"This code ran in {total_time/len(test_images)} seconds.")
print(f"Accuracy of this model is: {num_correct/len(test_images)}.")

