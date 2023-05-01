import cv2
import numpy as np
from PIL import Image
import scipy
import time

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def conv(input, filter):
    output = np.zeros(input.shape)
    for row in range(output.shape[0]): # adjusting for padding and size of filter
        for col in range(output.shape[1]): # adjusting for padding and size of filter
            sum = 0
            for kh in range(filter.shape[0]):
                for kw in range(filter.shape[1]):
                    r = row - filter.shape[0] // 2 + kh
                    c = col - filter.shape[1] // 2 + kw
                    if r >= 0 and c >= 0 and r < output.shape[0] and c < output.shape[1]:
                        sum += input[r][c] * filter[kh][kw]
            output[row][col] = sum
                
    return output.astype(float)


img = cv2.imread('irongiant.png',0)
start = time.perf_counter()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
big_filter = np.ones(shape=(30,30))/(30*30)

g = np.fft.fft2(big_filter, fshift.shape)
F_gaussian = np.fft.fftshift(g)

F_filtered_img = fshift*F_gaussian
filtered_img = np.fft.ifft2(np.fft.ifftshift(F_filtered_img)).real

print(f"Single layer FFT Convolution (30x30 filter): Process took {time.perf_counter()-start} seconds.")

start = time.perf_counter()
x = scipy.signal.convolve(img, big_filter, method='direct', mode='same')
print(f"Single layer Direct Convolution (30x30 filter): Process took {time.perf_counter()-start} seconds.")


img = cv2.imread('irongiant.png',0)
start = time.perf_counter()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

medium_filter = np.ones(shape=(15,15))/(15*15)

g = np.fft.fft2(medium_filter, fshift.shape)
F_gaussian = np.fft.fftshift(g)

F_filtered_img = fshift*F_gaussian
filtered_img = np.fft.ifft2(np.fft.ifftshift(F_filtered_img)).real
print(f"Single layer FFT Convolution (15x15 filter): Process took {time.perf_counter()-start} seconds.")

start = time.perf_counter()
x = scipy.signal.convolve(img, medium_filter, method='direct', mode='same')
print(f"Single layer Direct Convolution (15x15 filter): Process took {time.perf_counter()-start} seconds.")

img = cv2.imread('irongiant.png',0)
start = time.perf_counter()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

small_filter = np.ones(shape=(3,3))/(3*3)

g = np.fft.fft2(small_filter, fshift.shape)
F_gaussian = np.fft.fftshift(g)

F_filtered_img = fshift*F_gaussian
filtered_img = np.fft.ifft2(np.fft.ifftshift(F_filtered_img)).real
print(f"Single layer FFT Convolution (3x3 filter): Process took {time.perf_counter()-start} seconds.")

start = time.perf_counter()
x = scipy.signal.convolve(img, small_filter, method='direct', mode='same')
print(f"Single layer Direct Convolution (3x3 filter): Process took {time.perf_counter()-start} seconds.")