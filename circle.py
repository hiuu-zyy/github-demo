import cv2
import numpy as np

# Đọc hình ảnh
image = cv2.imread('IMAGE.jpeg')
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
    minRadius=1, 
    maxRadius=30
)

# Nếu phát hiện được ít nhất một hình tròn
if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Lấy giá trị y trên và dưới của tất cả các hình tròn
    y_top_values = circles[0, :, 1] - circles[0, :, 2]
    y_bottom_values = circles[0, :, 1] + circles[0, :, 2]
    
    # Tìm giá trị y nhỏ nhất và lớn nhất từ viền trên và viền dưới của các hình tròn
    min_y = np.min(y_top_values)
    max_y = np.max(y_bottom_values)
    dist = max_y - min_y
    ratio_img = dist/50 # assume that width of reel = 50mm
    l1 = min_y - int(ratio_img*0.7)
    l4 = max_y + int(ratio_img*0.7)
    l2 = min_y + int(ratio_img*5)
    l3 = max_y - int(ratio_img*5)
    
    print(f"Giá trị y trên nhỏ nhất: {min_y}")
    print(f"Giá trị y dưới lớn nhất: {max_y}")
    
    # Vẽ tất cả các hình tròn
    for i in circles[0, :]:
        # Vẽ hình tròn
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 3)
        # # Vẽ tâm của hình tròn
        # cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        
    # Vẽ đường thẳng ngang đi qua y-min và y-max
    height, width = image.shape[:2]
    cv2.line(image, (0, l1), (width, l1), (0, 255, 0), 3)
    cv2.line(image, (0, l4), (width, l4), (0, 255, 0), 3)
    cv2.line(image, (0, l3), (width, l3), (0, 255, 0), 3)
    cv2.line(image, (0, l2), (width, l2), (0, 255, 0), 3)
# Hiển thị kết quả
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
