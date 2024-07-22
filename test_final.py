import cv2
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression

start_time = time.time()

def draw_hough_lines(image, lines, mask):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if mask[y1, x1] == 255 and mask[y2, x2] == 255:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Hàm để kiểm tra xem đường cong có nằm giữa hai đường thẳng không
def is_curve_between_lines(y_curve, y_line_top, y_line_bottom):
    return np.all((y_curve > y_line_top) & (y_curve < y_line_bottom))

# Hàm để fit polynomial
def poly_fit(x, y, degree=1):
    p = np.polyfit(x, y, degree)
    return np.poly1d(p)

def fit_polynomial_curve(image, masked, mask, width,y_start_upper_top, y_start_upper_bottom, y_start_lower_top, y_start_lower_bottom, y_end_upper_top, y_end_upper_bottom, y_end_lower_top, y_end_lower_bottom, degree=2):
# def fit_polynomial_curve(image, y_offset, width, degree=2):
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Làm mờ hình ảnh để giảm nhiễu
    gray_blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray_blurred, 50, 100)

    # Sử dụng Hough Line Probabilistic
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=100, minLineLength=50, maxLineGap=15)
    # # Vẽ các đường thẳng lên hình ảnh gốc
    # draw_hough_lines(mask, lines)

    # # Hiển thị kết quả
    # cv2.imshow('Detected Lines', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result = False
    # KHÔNG CẦN SPLINE
    if lines is not None:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # y1 = y1 + y_offset  add offset to fit the original image
            # y2 = y2 + y_offset
            if mask[y1, x1] == 255 and mask[y2, x2] == 255 and abs(x1-x2)>50:
                # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # if abs(x1-x2)>50:
                points.append([x1, y1])
                points.append([x2, y2])
        
        points = np.array(points)
        if len(points) == 0:
            return False
        else:
            x = points[:, 0]
            y = points[:, 1]
        
        # Fit polynomial
        poly = poly_fit(x, y, degree=2)

        # Vẽ đường cong fitted
        # width = image.shape[1]
        x_extended = np.linspace(0, width, num=width)
        y_extended = poly(x_extended)
        # Kiểm tra và vẽ các đường thẳng song song
        for i in range(len(x_extended) - 1):
            pt1 = (int(x_extended[i]), int(y_extended[i]))
            pt2 = (int(x_extended[i + 1]), int(y_extended[i + 1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 3)

        y_upper_top = np.linspace(y_start_upper_top,y_end_upper_top,width)
        y_upper_bottom = np.linspace(y_start_upper_bottom,y_end_upper_bottom,width)
        y_lower_top = np.linspace(y_start_lower_top,y_end_lower_top,width)
        y_lower_bottom = np.linspace(y_start_lower_bottom,y_end_lower_bottom,width)

        
        if is_curve_between_lines(y_extended, y_upper_top, y_upper_bottom):
            result = True
            cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 0, 255), 3)
            cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 0, 255), 3)
        
        elif is_curve_between_lines(y_extended, y_lower_top, y_lower_bottom):
            result = True
            cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
            cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)
        
        


    return result


# def detect_fit_line(roi, original_y_start):
#     edges_roi = cv2.Canny(roi, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
#     if lines is not None:
#         points = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#             if length > 50:
#                 points.append([x1, y1 + original_y_start])
#                 points.append([x2, y2 + original_y_start])
#         points = np.array(points)
#         X = points[:, 0].reshape(-1, 1)
#         y = points[:, 1]
#         reg = LinearRegression().fit(X, y)
#         y_fit_start = int(reg.predict(np.array([[0]]))[0])
#         y_fit_end = int(reg.predict(np.array([[width]]))[0])
#         cv2.line(image, (0, y_fit_start), (width, y_fit_end), (0, 0, 0), 3)

# Hàm để tạo mặt nạ từ các điểm đỉnh của hình bình hành
def create_parallelogram_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask

def detect_lines(image, dist):
    if isinstance(image, str):
        image = cv2.imread(image)
    else: image = image
    

    ##################################    DRAW ROI     #############################################

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
            y_start_upper_bottom = y_start_upper + dist
            y_end_upper_bottom = y_end_upper + dist


            points1 = np.array([[0, y_start_upper_top], [width, y_end_upper_top], [width, y_end_upper_bottom], [0, y_start_upper_bottom]])
            mask1 = create_parallelogram_mask(image, points1)
            masked1 = cv2.bitwise_and(image, image, mask=mask1)

            cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 0, 255), 3)
            cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 0, 255), 3)
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
            y_start_lower_top = y_start_lower - dist
            y_end_lower_top = y_end_lower - dist
            y_start_lower_bottom = y_start_lower #+ offset1
            y_end_lower_bottom = y_end_lower #+ offset1



            points2 = np.array([[0, y_start_lower_top], [width, y_end_lower_top], [width, y_end_lower_bottom], [0, y_start_lower_bottom]])
            mask2 = create_parallelogram_mask(image, points2)
            masked2 = cv2.bitwise_and(image, image, mask=mask2)

            cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
            cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)
        
        # Tạo các vùng giữa các đường thẳng song song
        y_max_lower = max(y_start_lower_top, y_end_lower_top)
        y_min_lower = max(y_start_lower_bottom, y_end_lower_bottom)

        y_max_upper = min(y_start_upper_top, y_end_upper_top)
        y_min_upper = min(y_start_upper_bottom, y_end_upper_bottom)

        roi_upper = gray_blurred[y_max_upper:y_min_upper, :]
        roi_lower = gray_blurred[y_max_lower:y_min_lower, :]

        # Áp dụng Hough Line và hồi quy tuyến tính trên các vùng ROI
        state1 = fit_polynomial_curve(image, masked1, mask1, width, y_start_upper_top, y_start_upper_bottom, y_start_lower_top, y_start_lower_bottom, y_end_upper_top, y_end_upper_bottom, y_end_lower_top, y_end_lower_bottom)
        state2 = fit_polynomial_curve(image, masked2, mask2, width, y_start_upper_top, y_start_upper_bottom, y_start_lower_top, y_start_lower_bottom, y_end_upper_top, y_end_upper_bottom, y_end_lower_top, y_end_lower_bottom)



        if state1:
            cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 255, 0), 3)
            cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 255, 0), 3)
        if state2:
            cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 255, 0), 3)
            cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 255, 0), 3)

        if state1 and state2:
            return True, image
        else:
            return False, image



# def process_folder(input_folder, output_folder, dist):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
#             is_anom, processed_image = detect_lines(image, dist)
            
#             output_path = os.path.join(output_folder, f"output_{filename}")
#             cv2.imwrite(output_path, processed_image)
            
#             print(f"{filename}: Anomaly Detected - {is_anom}")

start_time = time.time()
# input_folder = 'IMAGES'
# output_folder = 'CHECK'
dist = 50

# process_folder(input_folder, output_folder, dist)
image = cv2.imread('IMAGES2/image (1).png')
detect_lines(image, dist)
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")

# Hiển thị kết quả
cv2.imshow('Detected Lines with Fitted Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

