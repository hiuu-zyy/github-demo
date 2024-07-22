import cv2
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression

start_time = time.time()

def detect_lines(image, dist):
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w, _ = image.shape

    width = image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.medianBlur(gray, 3)

    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=10, 
        param1=50, 
        param2=25, 
        minRadius=1, 
        maxRadius=25
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        height = image.shape[0]
        upper_centers = []
        lower_centers = []
        
        for circle in circles[0, :]:
            x, y, r = circle
            if y < height / 2:
                upper_centers.append(circle)
            else:
                lower_centers.append(circle)
        
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

        def find_largest_group(groups):
            largest_group = max(groups, key=len)
            return np.array(largest_group) if len(largest_group) > 0 else None

        upper_groups = group_circles_by_y(upper_centers)
        lower_groups = group_circles_by_y(lower_centers)

        largest_upper_group = find_largest_group(upper_groups)
        largest_lower_group = find_largest_group(lower_groups)

        if largest_upper_group is not None:
            X_upper = largest_upper_group[:, 0].reshape(-1, 1)
            y_upper = largest_upper_group[:, 1]
            reg_upper = LinearRegression().fit(X_upper, y_upper)
            
            y_start_upper = int(reg_upper.predict(np.array([[0]]))[0])
            y_end_upper = int(reg_upper.predict(np.array([[width]]))[0])
            
            y_start_upper_top = y_start_upper
            y_end_upper_top = y_end_upper
            y_start_upper_bottom = y_start_upper + dist
            y_end_upper_bottom = y_end_upper + dist


            #### DRAW RED LINE FOR THE FIRST TIME, HAVENOT DONE ANYTHING YET #####
            # cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 0, 255), 3)
            # cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 0, 255), 3)

        if largest_lower_group is not None:
            X_lower = largest_lower_group[:, 0].reshape(-1, 1)
            y_lower = largest_lower_group[:, 1]
            reg_lower = LinearRegression().fit(X_lower, y_lower)
            
            y_start_lower = int(reg_lower.predict(np.array([[0]]))[0])
            y_end_lower = int(reg_lower.predict(np.array([[width]]))[0])
            
            y_start_lower_top = y_start_lower - dist
            y_end_lower_top = y_end_lower - dist
            y_start_lower_bottom = y_start_lower
            y_end_lower_bottom = y_end_lower

            # cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
            # cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)

        y_max_lower = max(y_start_lower_top, y_end_lower_top)
        y_min_lower = max(y_start_lower_bottom, y_end_lower_bottom)

        y_max_upper = min(y_start_upper_top, y_end_upper_top)
        y_min_upper = min(y_start_upper_bottom, y_end_upper_bottom)

    line_number_up = 0
    line_number_down = 0

    gray = 255 - gray
    edges = cv2.Canny(gray, threshold1=80, threshold2=100)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=100, minLineLength=100, maxLineGap=15)
    is_anom = False

    if lines is not None:
        grouped_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y0 = int((y1 + y2) / 2)
            found_group = False
            if (y0 > y_max_upper + 5 and y0 < y_min_upper - 5) or (y0 > y_max_lower + 5 and y0 < y_min_lower - 15):
                for group in grouped_lines:
                    gline = group[0]
                    gy = gline[-1]
                    if np.abs(gy - y0) < 40:
                        group.append((x1, x2, y0))
                        found_group = True
                        break
                if not found_group:
                    grouped_lines.append([(x1, x2, y0)])

        longest_lines = []
        for group in grouped_lines:
            lengths = []
            for line in group:
                x1, x2, y0 = line
                length = np.abs(x1 - x2)
                lengths.append(length)
            if lengths:
                longest_line_index = np.argmax(lengths)
                longest_lines.append(group[longest_line_index])

        for line in longest_lines:
            x1, x2, y0 = line

            # IF THE LINE DETECTED, DONT DRAW ANY LINE
            # ELSE DRAW RED LINE
            if (y0 > (y_max_upper) and y0 < (y_min_upper - 10)) :
                line_number_up += 1



            elif (y0 > (y_max_lower) and y0 < (y_min_lower - 15)):
                # cv2.line(image, (0, y0), (w, y0), (255, 0, 0), 4)
                # cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
                # cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)
                line_number_down += 1

            # else :
            #     # cv2.line(image, (0, y0), (w, y0), (255, 0, 0), 4)
            #     cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 0, 255), 3)
            #     cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 0, 255), 3)
            #     cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
            #     cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)

    if line_number_down < 1 or line_number_up < 1:
        is_anom = True
        cv2.line(image, (0, y_start_upper_top), (width, y_end_upper_top), (0, 0, 255), 3)
        cv2.line(image, (0, y_start_upper_bottom), (width, y_end_upper_bottom), (0, 0, 255), 3)
        cv2.line(image, (0, y_start_lower_top), (width, y_end_lower_top), (0, 0, 255), 3)
        cv2.line(image, (0, y_start_lower_bottom), (width, y_end_lower_bottom), (0, 0, 255), 3)

    return is_anom, image

def process_folder(input_folder, output_folder, dist):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            is_anom, processed_image = detect_lines(image, dist)
            
            output_path = os.path.join(output_folder, f"output_{filename}")
            cv2.imwrite(output_path, processed_image)
            
            print(f"{filename}: Anomaly Detected - {is_anom}")

start_time = time.time()
input_folder = 'IMAGES'
output_folder = 'CHECKS'
dist = 55

process_folder(input_folder, output_folder, dist)

end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds")
