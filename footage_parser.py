from itertools import combinations
import cv2
import numpy as np

def detect_blue_rectangle(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100, 150, 50])  # Lower bound for blue
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Store detected blue circles
    blue_circles = []

    for contour in contours:
        # Approximate the contour to a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Add the circle if it meets the size criteria
        if radius > 3:  # Minimum radius for detection
            blue_circles.append((int(x), int(y)))

    # Ensure at least 3 circles are detected
    if len(blue_circles) < 3:
        return None

    # Check for combinations of points forming a rectangle
    if len(blue_circles) >= 4:
        # Use the first 4 points that form a rectangle
        points = np.array(blue_circles[:4], dtype=np.float32)
    else:
        # Use 3 points and extrapolate the fourth
        points = np.array(blue_circles, dtype=np.float32)

        # Extrapolate the missing fourth point
        if len(points) == 3:
            p1, p2, p3 = points
            missing_point = p3 + (p2 - p1)  # Calculate using vector addition
            points = np.vstack([points, missing_point])

    # Fit a bounding rectangle
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Integer points for drawing

    return box  # Return the rectangle as 4 points

# Function to detect red circle
def detect_red_circle(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and check for the largest circle
    red_circle = None
    for contour in contours:
        # Approximate the contour to a circle
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check if the contour is circular (with some tolerance on the aspect ratio)
        if len(approx) > 3:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 3:  # Minimum radius for detection
                red_circle = (int(x), int(y), int(radius))
                break

    return red_circle

# Load the video
video_path = 'data\RES600X600_RAD200_RATE60DPS.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # End of video

    # Detect the red circle
    red_circle = detect_red_circle(frame)

    rectangle = detect_blue_rectangle(frame)

    if rectangle is not None:
        # Draw the square (using a rectangle around the circle)
        cv2.rectangle(frame, rectangle, (255, 255, 0), 2)  # Green square

    if red_circle:
        print(f"Red Circle detected at: {red_circle[:2]} with radius {red_circle[2]}")

        # Create a square around the red circle
        center_x, center_y, radius = red_circle
        top_left = (center_x - radius, center_y - radius)
        bottom_right = (center_x + radius, center_y + radius)

        # Draw the square (using a rectangle around the circle)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green square

    # Show the frame with the detected red circle and square
    cv2.imshow('Red Circle with Square', frame)

    # Break the loop if the user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
