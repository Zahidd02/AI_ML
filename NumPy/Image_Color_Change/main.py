from skimage.io import *
import matplotlib.pyplot as plt
import numpy as np
import sys

# To check full output of the array
np.set_printoptions(threshold=sys.maxsize)

# Open path to the image
path = r'C:\Users\zahid\source\repos\Machine Learning\NumPy\Image_Color_Change\astronaut.png'
img = imread(path)

# Show original image
print(img.shape)  # Prints the image matrix
plt.imshow(img)
plt.show()

# Create a temp image variable to modify
img_temp = np.copy(img)
# Select on part of image with dimensions [y1:y2, x1:x2]
img_temp = img_temp[0:300, 350:475]
plt.imshow(img_temp)
plt.show()

# Change all the black color in temp image to yellow.
img_temp[np.less_equal(img_temp[:, :, 0], 45) & np.less_equal(img_temp[:, :, 0], 45)
         & np.less_equal(img_temp[:, :, 2], 45)] = [255, 255, 0]
plt.imshow(img_temp)
plt.show()

# Replace the temp image over original image
img[0:300, 350:475] = img_temp
plt.imshow(img)
plt.show()

# To change the color of every black color in original image
img[np.less_equal(img[:, :, 0], 45) & np.less_equal(img[:, :, 1], 45)
    & np.less_equal(img[:, :, 2], 45)] = [255, 255, 0]
plt.imshow(img)
plt.show()
