import cv2

image = cv2.imread("./fw10annotatedopgs/imgs/001.jpg")

height, width = image.shape[:2]
scale_factor = 0.3
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
resized_image = cv2.resize(image, (new_width, new_height))

if image is None:
    print("Error: Unable to load image.")
else:
    cv2.imshow("Loaded Image", resized_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
