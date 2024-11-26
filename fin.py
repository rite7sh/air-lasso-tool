import numpy as np
import cv2
from collections import deque

# Function to update slider values (required for trackbars)
def update_slider(x):
    pass

# Initialize color settings window and compact sliders
cv2.namedWindow("Color Settings")
cv2.createTrackbar("Lower Hue", "Color Settings", 10, 180, update_slider)
cv2.createTrackbar("Upper Hue", "Color Settings", 30, 180, update_slider)
cv2.createTrackbar("Lower Saturation", "Color Settings", 150, 255, update_slider)
cv2.createTrackbar("Upper Saturation", "Color Settings", 255, 255, update_slider)
cv2.createTrackbar("Lower Value", "Color Settings", 200, 255, update_slider)
cv2.createTrackbar("Upper Value", "Color Settings", 255, 255, update_slider)

# Colors for drawing
drawing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
current_color_index = 0

# Points storage for each color
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# Index counters
blue_idx, green_idx, red_idx, yellow_idx = 0, 0, 0, 0

# Canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255  # White canvas

# Toggle options
toggle_drawing = False
toggle_hsv = False

# Open webcam
capture = cv2.VideoCapture(0)

# Font settings
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 1
meter_background_color = (0, 0, 0)
meter_text_color = (255, 255, 255)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get slider values
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Color Settings")
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Color Settings")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Color Settings")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Color Settings")
    lower_value = cv2.getTrackbarPos("Lower Value", "Color Settings")
    upper_value = cv2.getTrackbarPos("Upper Value", "Color Settings")

    # HSV color range
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

    # Create mask
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours and toggle_drawing:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(largest_contour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Append points to the active color's deque
            if current_color_index == 0:
                blue_points[blue_idx].appendleft(center)
            elif current_color_index == 1:
                green_points[green_idx].appendleft(center)
            elif current_color_index == 2:
                red_points[red_idx].appendleft(center)
            elif current_color_index == 3:
                yellow_points[yellow_idx].appendleft(center)

    # Draw on the canvas
    points = [blue_points, green_points, red_points, yellow_points]
    for i, color_points in enumerate(points):
        for point_deque in color_points:
            for k in range(1, len(point_deque)):
                if point_deque[k - 1] is None or point_deque[k] is None:
                    continue
                cv2.line(canvas, point_deque[k - 1], point_deque[k], drawing_colors[i], 2)
                cv2.line(frame, point_deque[k - 1], point_deque[k], drawing_colors[i], 2)

    # Merge frame and canvas
    combined_view = np.hstack((frame, canvas))

    # Drawing toggle meter
    cv2.rectangle(combined_view, (10, 10), (150, 50), meter_background_color, -1)
    status_text = "Drawing: ON" if toggle_drawing else "Drawing: OFF"
    cv2.putText(combined_view, status_text, (20, 40), font_face, font_scale, meter_text_color, font_thickness)

    # Display frames
    cv2.imshow("Paint and Camera", combined_view)
    if toggle_hsv:
        cv2.imshow("HSV Mask", mask)

    # Keyboard shortcuts
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("b"):  # Select blue
        current_color_index = 0
    elif key == ord("g"):  # Select green
        current_color_index = 1
    elif key == ord("r"):  # Select red
        current_color_index = 2
    elif key == ord("y"):  # Select yellow
        current_color_index = 3
    elif key == ord("c"):  # Clear canvas
        blue_points = [deque(maxlen=1024)]
        green_points = [deque(maxlen=1024)]
        red_points = [deque(maxlen=1024)]
        yellow_points = [deque(maxlen=1024)]
        canvas[:, :, :] = 255
        blue_idx, green_idx, red_idx, yellow_idx = 0, 0, 0, 0
    elif key == ord("h"):  # Toggle HSV mask
        toggle_hsv = not toggle_hsv
    elif key == ord("d"):  # Toggle drawing
        toggle_drawing = not toggle_drawing

# Release resources
capture.release()
cv2.destroyAllWindows()
