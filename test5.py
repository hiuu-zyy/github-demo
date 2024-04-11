import cv2
import numpy as np
import os
import time

start_time = time.time()

def detect_lines(image1,output_folder):

    global processed_images_count
    image = image1
    h,w,_ = image.shape
    cv2.line(image1, (0, 240), (w, 240), (0, 0, 255), 4)
    cv2.line(image1, (0, 320), (w, 320), (0, 0, 255), 4)
    cv2.line(image1, (0, 970), (w, 970), (0, 0, 255), 4)
    cv2.line(image1, (0, 1050), (w, 1050), (0, 0, 255), 4)
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
    state = True
    # Filter and keep the longest lines with similar gradients and intercepts
    if lines is not None:
        grouped_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # x0 = int((x1+x2)/2)
            y0 = int((y1+y2)/2)
            # intercept = y1 - gradient * x1
            found_group = False
            if (y0 > 250 and y0 < 310) or (y0>978 and y0 < 1025):
                for group in grouped_lines:
                    gline = group[0]
                    gy = gline[-1]
                    if np.abs(gy-y0) < 70 :
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
                cv2.line(image1, (0, y0), (w, y0), (255, 0, 0), 4)
                cv2.line(image1, (0, 240), (w, 240), (0, 255, 0), 4)
                cv2.line(image1, (0, 320), (w, 320), (0, 255, 0), 4)
                line_number_up += 1
            elif (y0>978 and y0 < 1025):
                cv2.line(image1, (0, y0), (w, y0), (255, 0, 0), 4)
                cv2.line(image1, (0, 970), (w, 970), (0, 255, 0), 4)
                cv2.line(image1, (0, 1050), (w, 1050), (0, 255, 0), 4)
                line_number_down += 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"output_image_{processed_images_count + 1}.png")
    cv2.imwrite(output_path, image1)
    if line_number_down < 1 or line_number_up < 1:
        state = False

    # increase number of processing image
    processed_images_count += 1
    return state

processed_images_count = 0
output_folder = 'Folder_name'
image = cv2.imread('Path_to_image')
STATE = detect_lines(image, output_folder)
print(STATE)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")



