from itertools import combinations
import math
import time
import cv2
import numpy as np

width = 1400
height = 800

def detect_corners(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Store detected blue circles
    green_circles = []

    for contour in contours:
        # Approximate the contour to a circle
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check if the contour is circular (with some tolerance on the aspect ratio)
        if len(approx) > 3:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Minimum radius for detection
                green_circle = (int(x), int(y), int(radius))
                green_circles.append(green_circle)

    return green_circles  # Return the rectangle as 4 points

# Function to detect red circle
def detect_wire(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and check for the largest circle
    wire_end = None
    for contour in contours:
        # Approximate the contour to a circle
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check if the contour is circular (with some tolerance on the aspect ratio)
        if len(approx) > 3:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Minimum radius for detection
                wire_end = (int(x), int(y), int(radius))
                break

    return wire_end

def map_point_to_normalized_space(rect_corners, point):
    # Unpack rectangle corners
    p1, p2, p3, p4 = rect_corners
    x1, y1 = p1[0]
    x2, y2 = p2[0]
    x3, y3 = p3[0]
    x4, y4 = p4[0]
    
    px, py = point
    
    # Define source and destination points
    src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    dst = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    
    # Corrected: Solve the transformation matrix
    A = []
    B = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        B.append(dx)
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        B.append(dy)
    A = np.array(A)
    B = np.array(B)
    # Solve for h (8 parameters)
    h = np.linalg.lstsq(A, B, rcond=None)[0]
    # Append 1 to h to make it a 9-element vector and reshape to 3x3
    H = np.append(h, 1).reshape(3, 3)
    
    # Map the point using the homography matrix
    mapped = np.dot(H, np.array([px, py, 1]))
    u, v = mapped[0] / mapped[2], mapped[1] / mapped[2]
    return u, v


def parse_video(video_path = 'data/28.11.24/robot_footage.mp4'):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    delta_time = 0
    wire_positions = []
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        wire_end = detect_wire(frame)
        corners = detect_corners(frame)
        if len(corners) == 4 and wire_end:
            points = np.array([[x,y] for x, y, _ in corners])
            centroid = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_points = points[np.argsort(angles)]
            sorted_points = sorted_points.reshape((-1, 1, 2))
            wire_end_pos = wire_end[0], wire_end[1]
            norm_pred_x, norm_pred_y = map_point_to_normalized_space(sorted_points, wire_end_pos)
            pred_x = norm_pred_x * width
            pred_y = norm_pred_y * height
            wire_positions.append({
                "time": delta_time, 
                "pos": [pred_x, pred_y]
            })
            
        delta_time += 1 / frame_rate

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    return wire_positions
