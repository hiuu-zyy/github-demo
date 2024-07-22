import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# Đọc hình ảnh
image_path = 'IMAGES2/image (7).png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ hình ảnh để giảm nhiễu
gray_blurred = cv2.medianBlur(gray, 3)

# Phát hiện các hình tròn sử dụng HoughCircles
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=10, 
    param1=50, 
    param2=27, 
    minRadius=1, 
    maxRadius=30
)

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

    def group_circles_by_y(circles, max_distance=10):
        groups = []
        for circle in circles:
            x, y, r = circle
            added_to_group = False
            for group in groups:
                if abs(group[0][1] - y) <= max_distance:
                    group.append(circle)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([circle])
        return groups

    upper_groups = group_circles_by_y(upper_centers)
    lower_groups = group_circles_by_y(lower_centers)
    
    # Vẽ tất cả các hình tròn
    for i in circles[0, :]:
        # Vẽ hình tròn
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Vẽ tâm của hình tròn
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
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
        
        # Tính và vẽ các đường thẳng song song cách nhau 20 pixel
        angle_upper = np.arctan2(y_end_upper - y_start_upper, width)
        offset = int(20 / np.cos(angle_upper))
        y_start_upper_top = y_start_upper - offset
        y_end_upper_top = y_end_upper - offset
        y_start_upper_bottom = y_start_upper + offset
        y_end_upper_bottom = y_end_upper + offset

        # cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (255, 0, 0), 3)
        # cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (255, 0, 0), 3)
    
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
        
        # Tính và vẽ các đường thẳng song song cách nhau 20 pixel
        angle_lower = np.arctan2(y_end_lower - y_start_lower, width)
        offset = int(20 / np.cos(angle_lower))
        y_start_lower_top = y_start_lower - offset
        y_end_lower_top = y_end_lower - offset
        y_start_lower_bottom = y_start_lower + offset
        y_end_lower_bottom = y_end_lower + offset

        # cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
        # cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)

# Hiển thị kết quả
cv2.imshow('Detected Circles with Fitted Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


