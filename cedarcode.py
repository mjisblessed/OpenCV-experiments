import cv2
import numpy as np
import os

def extract_signature(image_path, min_contour_area=500, padding=15):
    # Reading the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better handling of varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean noise and connect components
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)


    # Finding external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filter contours by minimum area and aspect ratio (optional, but can help filter non-signature elements)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if not valid_contours:
        return None

    # Get bounding boxes and compute the combined bounding box
    x_min = y_min = float('inf')
    x_max = y_max = 0

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Add padding and clamp to image dimensions
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, img.shape[1])
    y_max = min(y_max + padding, img.shape[0])

    # Crop the region
    signature = img[y_min:y_max, x_min:x_max]
    return signature


def main():
    input_folder = os.path.join('CEDAR', 'full_org')
    output_folder = os.path.join('CEDAR', 'signatures')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            signature = extract_signature(image_path)
            if signature is not None:
                out_path = os.path.join(output_folder, filename)
                cv2.imwrite(out_path, signature)
                print(f"Extracted signature saved to {out_path}")
            else:
                print(f"No signature found in {filename}")

if __name__ == "__main__":
    main()