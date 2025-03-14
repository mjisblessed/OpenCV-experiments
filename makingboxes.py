import cv2

# Load the image
image_path = "./fw10annotatedopgs/imgs/001.jpg"
image = cv2.imread(image_path)

# Annotation data (replace with your actual data)
data = [
    {
        "image": "./fw10annotatedopgs/imgs/001.jpg",
        "verified": False,
        "annotations": [
            {"label": "B", "coordinates": {"x": 1130.5, "y": 1526.5, "width": 85.0, "height": 79.0}},
            {"label": "A", "coordinates": {"x": 1894.0, "y": 1199.5, "width": 72.0, "height": 87.0}}
        ]
    }
]

# Define colors for each label
colors = {"A": (0, 255, 0), "B": (0, 0, 255)}  # Green for "A", Red for "B"
# Draw bounding boxes
for annotation in data[0]["annotations"]:
    label = annotation["label"]
    coords = annotation["coordinates"]
    
    # Calculate top-left and bottom-right coordinates
    # (if the coordinates specify the top-left corner)
    # x1 = int(coords["x"])
    # y1 = int(coords["y"])
    # x2 = x1 + int(coords["width"])
    # y2 = y1 + int(coords["height"])
    
    # (since the coordinates specify the center of the box)
    x1 = int(coords["x"] - coords["width"] / 2)
    y1 = int(coords["y"] - coords["height"] / 2)
    x2 = int(coords["x"] + coords["width"] / 2)
    y2 = int(coords["y"] + coords["height"] / 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], thickness=3)

    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[label], 2)

cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL) 
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()