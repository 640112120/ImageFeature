import cv2
import numpy as np
# Load the image as grayscale
img_gray = cv2.imread('abc.jpeg', 0)
win_size = img_gray.shape
cell_size = (8, 8)
block_size = (16, 16)
block_stride = (8, 8)
num_bins = 9
# Set the parameters of the HOG descriptor using the variablesdefined above
hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
cell_size, num_bins)
# Compute the HOG Descriptor for the gray scale image
hog_descriptor = hog.compute(img_gray)
print ('HOG Descriptor:', hog_descriptor)