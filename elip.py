import cv2
import numpy as np

def is_ellipse(contour):
    # Tính toán hình chữ nhật bao quanh contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Kiểm tra tỷ lệ khung hình để xác định có phải là hình elip hay không
    if 0.5 < aspect_ratio < 2:
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area
        if 0.6 < extent < 1.0:
            return True
    return False

# Đọc ảnh
image = cv2.imread('2024-04-29 17:17:43.478774_affined.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Áp dụng bộ lọc cạnh Canny
edges = cv2.Canny(blur, 50, 150)

# Tìm các đường viền trong ảnh
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Lọc và vẽ các hình elip
for contour in contours:
    if len(contour) >= 5 and is_ellipse(contour):  # fitEllipse cần ít nhất 5 điểm và kiểm tra hình elip
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# Hiển thị ảnh với các hình elip phát hiện được
cv2.imshow('Detected Ellipses', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
