import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = 'lineDetectionTest/2024-04-29 12:02:00.094034_affined.png'





image = cv2.imread(image_path)



# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
gray = 255-gray
# Apply Canny edge detection
edges = cv2.Canny(gray, threshold1=70, threshold2=80)

cv2.imshow('Detected Edges', edges)


# cv2.imshow('Detected Edges', edges)

# Đợi phím bấm
cv2.waitKey(0)

# Đóng tất cả các cửa sổ hiển thị
cv2.destroyAllWindows() # type: ignore

# Hieu Hieu