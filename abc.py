import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = 'IMG (16).jpg'





image = cv2.imread(image_path)
h,w,_ = image.shape
image[320:975,:] = 0


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
gray = 255-gray
# Apply Canny edge detection
edges = cv2.Canny(gray, threshold1=80, threshold2=100)
scale_percent = 50  # tỉ lệ mới so với kích thước gốc
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
# resized_image = cv2.resize(image, (width, height))

# Thay đổi tỉ lệ của cạnh để phù hợp với màn hình
resized_edges = cv2.resize(edges, (width, height))

# Hiển thị hình ảnh gốc và các cạnh
# cv2.imshow('Original Image', resized_image)
cv2.imshow('Detected Edges', resized_edges)


# cv2.imshow('Detected Edges', edges)

# Đợi phím bấm
cv2.waitKey(0)

# Đóng tất cả các cửa sổ hiển thị
cv2.destroyAllWindows() # type: ignore

# Hieu Hieu