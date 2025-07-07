import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_signature(image_path, min_contour_area=50, padding=15):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    h_img, w_img = img.shape[:2]
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(0)

    # Histogram debug (optional, comment out later)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title("Grayscale Histogram")
    plt.show()

    # Otsu Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Thresholded (Otsu)", thresh)
    cv2.waitKey(0)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("Cleaned", cleaned)
    cv2.waitKey(0)

    # Contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h_img * w_img

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 0.98 * img_area:  # reject near-full-image blobs
            valid_contours.append(cnt)


    print(f"Valid contours (area > {min_contour_area}): {len(valid_contours)}")

    # Draw all contours (for debugging)
    debug_contours = img.copy()
    cv2.drawContours(debug_contours, contours, -1, (255, 0, 0), 1)
    cv2.imshow("All Contours", debug_contours)
    cv2.waitKey(0)

    if not valid_contours:
        print("No valid contours found after area filtering.")
        cv2.destroyAllWindows()
        return None

    # Find bounding box
    x_min = y_min = float('inf')
    x_max = y_max = 0
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, w_img)
    y_max = min(y_max + padding, h_img)

    bbox_w, bbox_h = x_max - x_min, y_max - y_min

    print(f"Bounding box: x={x_min}, y={y_min}, w={bbox_w}, h={bbox_h}")
    print(f"Original size: {w_img}x{h_img}")

    if bbox_w == w_img and bbox_h == h_img:
        print("Extracted region is exactly the original image — skipping.")
        cv2.destroyAllWindows()
        return None

    # Optional: Draw bounding box
    boxed_img = img.copy()
    cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Bounding Box", boxed_img)
    cv2.waitKey(0)

    signature = img[y_min:y_max, x_min:x_max]
    cv2.imshow("Extracted Signature", signature)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return signature


def main():
    input_folder = os.path.join('CEDAR', 'practice')
    output_folder = os.path.join('CEDAR', 'signatures')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            print(f"\nProcessing: {filename}")
            signature = extract_signature(image_path)
            if signature is not None:
                out_path = os.path.join(output_folder, filename)
                cv2.imwrite(out_path, signature)
                print(f"Extracted signature saved to {out_path}")
            else:
                print(f"No valid signature extracted from {filename}")

if __name__ == "__main__":
    main()
