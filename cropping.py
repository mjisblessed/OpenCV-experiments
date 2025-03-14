import cv2
import numpy as np

# Load the image
image = cv2.imread("./fw10annotatedopgs/imgs/001.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary image (assuming white background)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the darker central region)
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding box of the largest dark region
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image
cropped_image = image[y:y+h, x:x+w]

# Example: Transform an old coordinate (x_old, y_old)
x_old, y_old = 500, 300  
if x <= x_old <= x + w and y <= y_old <= y + h:
    x_new, y_new = x_old - x, y_old - y
    print(f"New coordinates: ({x_new}, {y_new})")
else:
    print("Old coordinate is outside the cropped region.")

cv2.imwrite("cropped_image.jpg", cropped_image)
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
