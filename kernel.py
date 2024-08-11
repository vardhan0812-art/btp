import cv2

# Read the image
img = cv2.imread('image.jpg')

# Save the image to a file
cv2.imwrite('output_image.jpg', img)

# Optionally, open the image using an external viewer
import os
os.system('output_image.jpg')
