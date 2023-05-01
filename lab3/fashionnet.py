import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from fashionnet_utils import FFTConv2D, FullyConnected, Flatten, Conv2D, MaxPooling

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
data_path = "./fashionnet/"
fnconv1_w = np.load(data_path + "fnconv1_w.npy")
fnconv2_w = np.load(data_path + "fnconv2_w.npy")
fnconv1_b = np.load(data_path + "fnconv1_b.npy")
fnconv2_b = np.load(data_path + "fnconv2_b.npy")
fc1_w = np.load(data_path + "fc1_w.npy")
fc1_b = np.load(data_path + "fc1_b.npy")
fc2_w = np.load(data_path + "fc2_w.npy")
fc2_b = np.load(data_path + "fc2_b.npy")
fc3_w = np.load(data_path + "fc3_w.npy")
fc3_b = np.load(data_path + "fc3_b.npy")

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
    print(guess,out)
    if(guess == test_labels[i]):
        count += 1

print(count/100)
