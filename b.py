import cv2
import numpy as np

# Hàm để fit polynomial
def poly_fit(x, y, degree=2):
    p = np.polyfit(x, y, degree)
    return np.poly1d(p)

# Đọc hình ảnh
image_path = 'IMAGES2/image (6).png'
image = cv2.imread(image_path)
image = image[247:305, :]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ hình ảnh để giảm nhiễu
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Phát hiện cạnh bằng phương pháp Canny
edges = cv2.Canny(gray_blurred, 50, 150, apertureSize=3)

# Phát hiện các đoạn thẳng sử dụng HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

if lines is not None:
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append([x1, y1])
        points.append([x2, y2])
    
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Fit polynomial
    poly = poly_fit(x, y, degree=2)

    # Vẽ đường cong fitted
    width = image.shape[1]
    x_extended = np.linspace(0, width, num=width)
    y_extended = poly(x_extended)
    
    for i in range(len(x_extended) - 1):
        pt1 = (int(x_extended[i]), int(y_extended[i]))
        pt2 = (int(x_extended[i + 1]), int(y_extended[i + 1]))
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

# Hiển thị kết quả
cv2.imshow('Detected Curves', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu hình ảnh kết quả
cv2.imwrite('/mnt/data/Detected_Curves.png', image)
