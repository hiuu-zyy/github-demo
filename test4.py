import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import time

start_time = time.time()

def detect_lines_and_circles(image_path,output_path):
    # Load the image
    image1 = cv2.imread(image_path)
    image = cv2.imread(image_path)
    h,w,_ = image.shape

    line_number_up = 0
    line_number_down = 0


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
    gray = 255-gray
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=80, threshold2=100)

    # Detect lines using HoughLinesP on the edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=100, minLineLength=100, maxLineGap=10)

    # Filter and keep the longest lines with similar gradients and intercepts
    if lines is not None:
        grouped_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # x0 = int((x1+x2)/2)
            y0 = int((y1+y2)/2)
            # intercept = y1 - gradient * x1
            found_group = False
            if (y0 > 250 and y0 < 310) or (y0>984 and y0 < 1025):
                for group in grouped_lines:
                    gline = group[0]
                    gy = gline[-1]
                    if np.abs(gy-y0) < 50 :
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
                length = np.abs(x1-x2)
                lengths.append(length)
            if lengths:  # Check if lengths is not empty
                longest_line_index = np.argmax(lengths)
                longest_lines.append(group[longest_line_index])

        # Draw the longest lines on the image
        for line in longest_lines:
            x1, x2, y0 = line
            if (y0 > 250 and y0 < 310)  : 
                cv2.line(image1, (0, y0), (w, y0), (0, 255, 0), 4)
                line_number_up += 1
            elif (y0>984 and y0 < 1025):
                cv2.line(image1, (0, y0), (w, y0), (0, 255, 0), 4)
                line_number_down += 1
    cv2.imwrite(output_path, image1)
    return line_number_up, line_number_down

def check_image_condition_in_folder(input_folder_path, output_folder_path):
    all_good = True
    num_images = 0
    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(input_folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                num_images += 1
                image_path = os.path.join(input_folder_path, filename)
                output_path = os.path.join(output_folder_path, f'check{num_images}.png')
                futures.append(executor.submit(detect_and_save_image, image_path, output_path))
        
        for future in futures:
            up, down, image_num = future.result()
            if up < 1 or down < 1:
                print(f"Image {image_num}: NG")
                all_good = False
    
    if all_good:
        print("ALL GOOD")

def detect_and_save_image(image_path, output_path):
    up, down = detect_lines_and_circles(image_path, output_path)
    image_num = int(output_path.split('.')[0].split('check')[-1])
    return up, down, image_num


input_folder_path = r"C:\Users\laVie\Documents\KakaoTalk Downloads\Test\IMAGES"
output_folder_path = r"C:\Users\laVie\Documents\KakaoTalk Downloads\Test\CheckedImages"

check_image_condition_in_folder(input_folder_path, output_folder_path)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# ngu
# dung sua nua

