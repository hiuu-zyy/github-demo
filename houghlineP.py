import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# Hàm để fit polynomial
def poly_fit(x, y, degree=2):
    p = np.polyfit(x, y, degree)
    return np.poly1d(p)

# Đọc ảnh
image_path = 'IMAGES/IMG (2).jpg'
image = cv2.imread(image_path)
image = image[990:1034, :]
width = image.shape[1]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Phát hiện cạnh
edges = cv2.Canny(gray, 50, 150)

# Sử dụng Hough Line Probabilistic
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Lưu trữ các điểm
points = []

# Lấy điểm từ các đoạn thẳng
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))
        # cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 2)

# Chuyển đổi điểm thành numpy array
points = np.array(points)

# Tìm đường cong (spline)
if len(points) > 3:  # Cần ít nhất 4 điểm để fit spline
    # Lấy các điểm duy nhất và sắp xếp theo thứ tự
    unique_points = np.unique(points, axis=0)
    unique_points = unique_points[np.argsort(unique_points[:, 0])]

    # Tạo spline
    tck, u = splprep(unique_points.T, s=5)
    x, y = splev(np.linspace(0, 1, 100), tck)
    
    # Fit đường polynomial
    poly = poly_fit(x, y, degree=2)

    # Vẽ đường cong fitted
    # for i in range(len(x) - 1):
    #     pt1 = (int(x[i]), int(poly(x[i])))
    #     pt2 = (int(x[i + 1]), int(poly(x[i + 1])))
        # cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Đường polynomial màu xanh lá
            # Ngoại suy để vẽ đường cong từ 0 đến width
    x_extended = np.linspace(0, width, num=width)
    y_extended = poly(x_extended)
    for i in range(len(x_extended) - 1):
        pt1 = (int(x_extended[i]), int(y_extended[i]))
        pt2 = (int(x_extended[i + 1]), int(y_extended[i + 1]))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

# Hiển thị ảnh kết quả
cv2.imshow('Fitted Curve', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
