import cv2
import numpy as np
from scipy.optimize import curve_fit

# Hàm để fit polynomial
def poly_fit(x, y, degree=2):
    p = np.polyfit(x, y, degree)
    return np.poly1d(p)

# Đọc hình ảnh
image_path = 'IMAGES/IMG (2).jpg'
image = cv2.imread(image_path)
image = image[255:320, :]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ hình ảnh để giảm nhiễu
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Phát hiện cạnh bằng phương pháp Canny
edges = cv2.Canny(gray_blurred, 50, 70, apertureSize=3)

# Tìm các đường cong (contours)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ các contours
cv2.drawContours(image, contours, -1, (255, 0, 0), 1)  # Vẽ contours bằng màu xanh lam

# Vẽ các đường polynomial cho contour có dạng cong
for contour in contours:
    if len(contour) > 400:  # Chỉ xử lý những đường cong đủ dài
        # Tính độ cong (có thể thay đổi theo ý bạn)
        arc_length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (arc_length ** 2) if arc_length > 0 else 0
        
        # Kiểm tra nếu contour có dạng cong (circularity < 0.5 có thể điều chỉnh)
        if circularity < 0.5:
            # Lấy các điểm (x, y)
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]

            # Fit polynomial
            poly = poly_fit(x, y, degree=2)

            # Vẽ đường cong fitted
            for i in range(len(x)):
                pt1 = (x[i], int(poly(x[i])))
                pt2 = (x[(i + 1) % len(x)], int(poly(x[(i + 1) % len(x)])))
                # cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Đường polynomial màu xanh lá

# Hiển thị kết quả
cv2.imshow('Detected Curves', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

