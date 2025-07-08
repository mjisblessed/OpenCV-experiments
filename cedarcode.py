import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_signature(image_path, min_contour_area=50, padding=15):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    h_img, w_img = img.shape[:2]

    # STEP 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 3: Apply Otsu's thresholding to segment foreground
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # STEP 4: Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # STEP 5: Find contours from the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h_img * w_img

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < 0.98 * img_area:
            valid_contours.append(cnt)

    print(f"Valid contours (area > {min_contour_area}): {len(valid_contours)}")

    # STEP 6: Draw all contours for visual debugging
    debug_contours = img.copy()
    cv2.drawContours(debug_contours, contours, -1, (255, 0, 0), 1)

    # Return early if no valid contours
    if not valid_contours:
        print("No valid contours found after area filtering.")
        return None

    # STEP 7: Compute bounding box covering all valid contours
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
        return None

    # STEP 8: Draw bounding box on image
    boxed_img = img.copy()
    cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # STEP 9: Extract the signature region
    signature = img[y_min:y_max, x_min:x_max]

    # STEP 10: Visualize all steps using matplotlib
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")

    axs[1].imshow(gray, cmap='gray')
    axs[1].set_title("Grayscale")

    axs[2].imshow(thresh, cmap='gray')
    axs[2].set_title("Otsu Threshold")

    axs[3].imshow(cleaned, cmap='gray')
    axs[3].set_title("Morphologically Cleaned")

    axs[4].imshow(cv2.cvtColor(debug_contours, cv2.COLOR_BGR2RGB))
    axs[4].set_title("All Contours")

    axs[5].imshow(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB))
    axs[5].set_title("Detected Signature Box")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

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
