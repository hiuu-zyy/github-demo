import cv2
import numpy as np

# Đọc hình ảnh
image = cv2.imread('IMAGE.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ hình ảnh để giảm nhiễu
gray_blurred = cv2.medianBlur(gray, 5)

# Phát hiện các hình tròn sử dụng HoughCircles
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=20, 
    param1=50, 
    param2=30, 
    minRadius=1, 
    maxRadius=30
)

# Nếu phát hiện được ít nhất một hình tròn
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Vẽ hình tròn
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Vẽ tâm của hình tròn
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Hiển thị kết quả
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
