import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "path/to/your/image.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5, 5), 0.8)

_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

kernel = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

min_area = 150   
max_area = 400  

bounding_boxes = []
for i in range(1, num_labels):  
    x, y, w, h, area = stats[i]
    if min_area <= area <= max_area:
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

print(f"Amount of DNA origami: {len(bounding_boxes)}")

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.title("Mask")
plt.imshow(cleaned, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Bounding boxes")
plt.imshow(img_color)
plt.axis('off')


plt.tight_layout()
plt.show()
