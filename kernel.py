import matplotlib.pyplot as plt
import cv2

# Read the image
image = cv2.imread("image.jpg")

# Convert the image from BGR (OpenCV default) to RGB (Matplotlib expects RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.axis('off')  # Hide the axes
plt.show()
