import tflite_runtime.interpreter as tflite
import numpy as np
import time

test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# data preprocessing to normalize data
# test_images = test_images / 255.0

# reshape data
test_images = test_images[..., np.newaxis]
test_images = test_images.astype(np.float32)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="my_model_dynamic_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


num_correct = 0
total_time = 0
for i in range(len(test_images)):
    # Test the model on random input data.
    interpreter.set_tensor(input_details[0]['index'], test_images[i].reshape((1, 28, 28, 1)))
    
    # Run the model
    start_time = time.time()
    interpreter.invoke()
    total_time += time.time() - start_time

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == test_labels[i]: num_correct += 1

print(f"This code ran in {total_time/len(test_images)} seconds.")
print(f"Accuracy of this model is: {num_correct/len(test_images)}.")

