import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import splprep, splev
import time

# Hàm để fit polynomial
def poly_fit(x, y, degree=2):
    p = np.polyfit(x, y, degree)
    return np.poly1d(p)

def fit_polynomial_curve(roi, y_offset,width, degree=2):

    edges = cv2.Canny(roi, 50, 150)

    # Sử dụng Hough Line Probabilistic
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)


    # CẦN SPLINE
    # Lưu trữ các điểm
    # points = []

    # # Lấy điểm từ các đoạn thẳng
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         points.append((x1, y1))
    #         points.append((x2, y2))

    # # Chuyển đổi điểm thành numpy array
    # points = np.array(points)

    # # Fit đường polynomial
    # if len(points) > 3:  # Cần ít nhất 4 điểm để fit
    #     unique_points = np.unique(points, axis=0)
    #     unique_points = unique_points[np.argsort(unique_points[:, 0])]

    #     # Tạo spline
    #     tck, u = splprep(unique_points.T, s=5)
    #     x, y = splev(np.linspace(0, 1, 100), tck)

    #     # Fit đường polynomial
    #     poly = poly_fit(x, y, degree=2)

    #     # Vẽ đường cong fitted
    #     # for i in range(len(x) - 1):
    #     #     pt1 = (int(x[i]), int(poly(x[i]) + y_offset))
    #     #     pt2 = (int(x[i + 1]), int(poly(x[i + 1]) + y_offset))
    #     #     cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Đường polynomial màu xanh lá
    #     x_extended = np.linspace(0, width, num=width)
    #     y_extended = poly(x_extended) + y_offset
    #     for i in range(len(x_extended) - 1):
    #         pt1 = (int(x_extended[i]), int(y_extended[i]))
    #         pt2 = (int(x_extended[i + 1]), int(y_extended[i + 1]))
    #         cv2.line(image, pt1, pt2, (0, 255, 0), 3)


    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)


    # KHÔNG CẦN SPLINE
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y1 = y1 + y_offset
            y2 = y2 + y_offset
            points.append([x1, y1])
            points.append([x2, y2])
        
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit polynomial
        poly = poly_fit(x, y, degree=2)

        # Vẽ đường cong fitted
        # width = image.shape[1]
        x_extended = np.linspace(0, width, num=width)
        y_extended = poly(x_extended)
        
        for i in range(len(x_extended) - 1):
            pt1 = (int(x_extended[i]), int(y_extended[i]))
            pt2 = (int(x_extended[i + 1]), int(y_extended[i + 1]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 3)

def detect_fit_line(roi, original_y_start):
    edges_roi = cv2.Canny(roi, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 50:
                points.append([x1, y1 + original_y_start])
                points.append([x2, y2 + original_y_start])
        points = np.array(points)
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        reg = LinearRegression().fit(X, y)
        y_fit_start = int(reg.predict(np.array([[0]]))[0])
        y_fit_end = int(reg.predict(np.array([[width]]))[0])
        cv2.line(image, (0, y_fit_start), (width, y_fit_end), (0, 0, 0), 3)

# Đọc hình ảnh
start_time = time.time()
image_path = 'IMAGES/IMG (11).jpg'
image = cv2.imread(image_path)
width = image.shape[1]
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
    minRadius=10, 
    maxRadius=25
)

offset1 = 23
offset2 = 60

# Nếu phát hiện được ít nhất một hình tròn
if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Chia các tâm hình tròn thành hai nhóm (trên và dưới)
    height = image.shape[0]
    upper_centers = []
    lower_centers = []
    
    for circle in circles[0, :]:
        x, y, r = circle
        if y < height / 2:
            upper_centers.append(circle)
        else:
            lower_centers.append(circle)
    
    # Chuyển đổi thành mảng numpy
    upper_centers = np.array(upper_centers)
    lower_centers = np.array(lower_centers)
    
    # Vẽ tất cả các hình tròn
    # for i in circles[0, :]:
        # # Vẽ hình tròn
        # cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # # Vẽ tâm của hình tròn
        # cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # Thực hiện hồi quy tuyến tính cho các hình tròn phía trên
    if len(upper_centers) > 0:
        X_upper = upper_centers[:, 0].reshape(-1, 1)
        y_upper = upper_centers[:, 1]
        reg_upper = LinearRegression().fit(X_upper, y_upper)
        
        # Tính toán các giá trị y dự đoán từ x = 0 đến x = width
        width = image.shape[1]
        y_start_upper = int(reg_upper.predict(np.array([[0]]))[0])
        y_end_upper = int(reg_upper.predict(np.array([[width]]))[0])
        
        # Vẽ đường thẳng fit nhất cho nhóm trên
        # cv2.line(image, (0, y_start_upper), (width, y_end_upper), (255, 0, 0), 2)
        
        # Tính và vẽ các đường thẳng song song
        y_start_upper_top = y_start_upper #- offset1
        y_end_upper_top = y_end_upper #- offset1
        y_start_upper_bottom = y_start_upper + offset2
        y_end_upper_bottom = y_end_upper + offset2

        cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (255, 0, 0), 3)
        cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (255, 0, 0), 3)
    
    # Thực hiện hồi quy tuyến tính cho các hình tròn phía dưới
    if len(lower_centers) > 0:
        X_lower = lower_centers[:, 0].reshape(-1, 1)
        y_lower = lower_centers[:, 1]
        reg_lower = LinearRegression().fit(X_lower, y_lower)
        
        # Tính toán các giá trị y dự đoán từ x = 0 đến x = width
        y_start_lower = int(reg_lower.predict(np.array([[0]]))[0])
        y_end_lower = int(reg_lower.predict(np.array([[width]]))[0])
        
        # Vẽ đường thẳng fit nhất cho nhóm dưới
        # cv2.line(image, (0, y_start_lower), (width, y_end_lower), (0, 0, 255), 2)
        
        # Tính và vẽ các đường thẳng song song 
        y_start_lower_top = y_start_lower - offset2
        y_end_lower_top = y_end_lower - offset2
        y_start_lower_bottom = y_start_lower #+ offset1
        y_end_lower_bottom = y_end_lower #+ offset1

        cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (255, 0, 0), 3)
        cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (255, 0, 0), 3)
    
    # Tạo các vùng giữa các đường thẳng song song
    y_max_lower = max(y_start_lower_top, y_end_lower_top)
    y_max_lower = y_max_lower + 14
    y_min_lower = min(y_start_lower_bottom, y_end_lower_bottom)

    y_max_upper = max(y_start_upper_top, y_end_upper_top)
    y_min_upper = min(y_start_upper_bottom, y_end_upper_bottom)
    y_min_upper = y_min_upper - 15

    y_min_top = max(y_start_upper_bottom, y_end_upper_bottom)
    
    roi_upper = gray_blurred[y_max_upper:y_min_upper, :]
    roi_lower = gray_blurred[y_max_lower:y_min_lower, :]


            # cv2.line(image, (0, y_fit_start - offset), (width, y_fit_end - offset), (0, 255, 255), 1)
            # cv2.line(image, (0, y_fit_start + offset), (width, y_fit_end + offset), (0, 255, 255), 1)
    
    # Áp dụng Hough Line và hồi quy tuyến tính trên các vùng ROI
    fit_polynomial_curve(roi_upper, y_max_upper, width)
    fit_polynomial_curve(roi_lower, y_max_lower, width)
    print(y_max_upper, y_min_upper)
end_time = time.time()
execute_time = end_time - start_time
print('Execute time: {}s'.format(execute_time))

# Hiển thị kết quả
cv2.imshow('Detected Lines with Fitted Line', image)
cv2.imshow('roi_up', image[y_max_upper:y_min_upper, :])
cv2.imshow('roi_low', image[980:1034 , :])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu hình ảnh kết quả
# cv2.imwrite('/mnt/data/Detected_Lines_with_Fitted_Line.png', image)
