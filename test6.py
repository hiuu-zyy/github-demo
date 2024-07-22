import cv2
import numpy as np
import time
import os

start_time = time.time()

def detect_lines(image, l1, l2, l3, l4, output_path):
    if isinstance(image, str):
        image = cv2.imread(image)
    else: image = image

    h,w,_ = image.shape
    ratio = np.sqrt(h ** 2 + w ** 2) / 1434
    ratio_y = h/1190
    cv2.line(image, (0, l1), (w, l1), (0, 0, 255), 4)
    cv2.line(image, (0, l2), (w, l2), (0, 0, 255), 4)
    cv2.line(image, (0, l3), (w, l3), (0, 0, 255), 4)
    cv2.line(image, (0, l4), (w, l4), (0, 0, 255), 4)
    line_number_up = 0
    line_number_down = 0


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
    gray = 255-gray
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=80, threshold2=100)

    # Detect lines using HoughLinesP on the edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=100, minLineLength=100*ratio, maxLineGap=15*ratio)
    state = True
    # Filter and keep the longest lines with similar gradients and intercepts
    if lines is not None:
        grouped_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y0 = int((y1+y2)/2)
            found_group = False
            if (y0 > l1+15 and y0 < l2-5) or (y0 > l3+5 and y0 < l4-15):
                for group in grouped_lines:
                    gline = group[0]
                    gy = gline[-1]
                    if np.abs(gy-y0) < 40 :
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
            if (y0 > (l1+15) and y0 < (l2-10))  : 
                cv2.line(image, (0, y0), (w, y0), (255, 0, 0), 4)
                cv2.line(image, (0, l1), (w, l1), (0, 255, 0), 4)
                cv2.line(image, (0, l2), (w, l2), (0, 255, 0), 4)
                line_number_up += 1
            elif (y0 > (l3+10) and y0 < (l4-15)):
                cv2.line(image, (0, y0), (w, y0), (255, 0, 0), 4)
                cv2.line(image, (0, l3), (w, l3), (0, 255, 0), 4)
                cv2.line(image, (0, l4), (w, l4), (0, 255, 0), 4)
                line_number_down += 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"output_image.png")
    cv2.imwrite(output_path, image)
    if line_number_down < 1 or line_number_up < 1:
        state = False

    return state


output_folder = 'Folder_name'
image = cv2.imread('2024-04-29 17:17:43.478774_affined.png')
# resized_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
# STATE = detect_lines(image,35,90,601,634, output_folder)
STATE = detect_lines(image,15,84,575,638, output_folder)
print(STATE)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")



