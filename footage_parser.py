from typing import List
import cv2
import numpy as np
import pandas as pd

class FootageParser:
    def __init__(self, width = 1280, height = 720, margin = 100):
        self.width = width - 2 * margin
        self.height = height - 2 * margin

    def detect_corners(self, image):
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

    def detect_wire(self, image):
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

    def map_point_to_normalized_space(self, rect_corners, point):
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

    def wire_screenspace_to_gridspace(self, corners, wire_end):
        if len(corners) == 4 and wire_end:
            points = np.array([[x,y] for x, y, _ in corners])
            centroid = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_points = points[np.argsort(angles)]
            sorted_points = sorted_points.reshape((-1, 1, 2))
            wire_end_pos = wire_end[0], wire_end[1]
            norm_pred_x, norm_pred_y = self.map_point_to_normalized_space(sorted_points, wire_end_pos)
            pred_x = norm_pred_x * self.width
            pred_y = self.height - norm_pred_y * self.height
            return [pred_x, pred_y]
        return None

    def parse_video(self, video_path, show = False):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_time = 0
        positions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            wire_end = self.detect_wire(frame)
            corners = self.detect_corners(frame)
            pos = self.wire_screenspace_to_gridspace(corners, wire_end)
            positions.append(pos)
            total_time += 1 / frame_rate
            if show:                    
                timer_text = f"{int(total_time // 60):02}:{int(total_time % 60):02}"
                
                for corner in corners:
                    center_x, center_y, radius = corner
                    top_left = (center_x - radius, center_y - radius)
                    bottom_right = (center_x + radius, center_y + radius)
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                
                if wire_end:
                    # Create a square around the red circle
                    center_x, center_y, radius = wire_end
                    top_left = (center_x - radius, center_y - radius)
                    bottom_right = (center_x + radius, center_y + radius)

                    # Draw the square (using a rectangle around the circle)
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green square
                
                if len(corners) == 4:
                    points = np.array([[x,y] for x, y, _ in corners])
                    centroid = np.mean(points, axis=0)
                    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
                    sorted_points = points[np.argsort(angles)]
                    sorted_points = sorted_points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [sorted_points], isClosed=True, color=(255, 255, 0), thickness=2)
                
                cv2.putText(
                    frame,                        # Frame
                    timer_text,                   # Text to display
                    (10, 50),                     # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,     # Font
                    1,                            # Font scale
                    (0, 0, 255),                  # Color (BGR - green)
                    2,                            # Thickness
                    cv2.LINE_AA                   # Line type
                )
                
                if pos:
                    cv2.putText(
                        frame,                        # Frame
                        f"{float(round(pos[0])), float(round(pos[1]))}",
                        (10, 100),                    # Position (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,     # Font
                        1,                            # Font scale
                        (0, 0, 255),                  # Color (BGR - green)
                        2,                            # Thickness
                        cv2.LINE_AA                   # Line type
                    )
                
                cv2.imshow('Footage', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture object and close any open windows
        cap.release()
        cv2.destroyAllWindows()
        
        positions = pd.Series(positions)
        positions = positions.ffill().bfill()
        if positions.isnull().all():
            positions = np.zeros((len(positions), 2))
        positions = {i / frame_rate : pos for (i, pos) in enumerate(positions)}
        return positions