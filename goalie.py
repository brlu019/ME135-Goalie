import cv2
import numpy as np
import time
from collections import deque

def nothing(x):
    pass

def createTrackbars():
    cv2.namedWindow("Calibration")
    cv2.createTrackbar("L - H", "Calibration", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Calibration", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Calibration", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Calibration", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Calibration", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Calibration", 255, 255, nothing)

def dynamicHSVCalibration():
    l_h = cv2.getTrackbarPos("L - H", "Calibration")
    l_s = cv2.getTrackbarPos("L - S", "Calibration")
    l_v = cv2.getTrackbarPos("L - V", "Calibration")
    u_h = cv2.getTrackbarPos("U - H", "Calibration")
    u_s = cv2.getTrackbarPos("U - S", "Calibration")
    u_v = cv2.getTrackbarPos("U - V", "Calibration")

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    return lower_range, upper_range

def morphCleaning(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    return mask

def getGoalPositions(cap):
    goalDetected = False
    createTrackbars()

    print("Press 'c' to confirm goal positions.")

    while not goalDetected:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))
        sv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_range, upper_range = dynamicHSVCalibration()

        # Color mask to detect markers
        mask = cv2.inRange(sv, lower_range, upper_range)
        mask = morphCleaning(mask)

        # Find contours (markers)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_centers = []

        if len(contours) >= 2:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2] # Get the two largest contours

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100: # Min area to filter noise
                    M = cv2.moments(cnt) # Calculate marker centers
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        marker_centers.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # Draw detected marker centers

            if len(marker_centers) == 2:
                cv2.line(frame, marker_centers[0], marker_centers[1], (255, 0, 0), 2)
                cv2.putText(frame, f"Marker 1{marker_centers[0]}", (marker_centers[0][0]+10, marker_centers[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Marker 2{marker_centers[1]}", (marker_centers[1][0]+10, marker_centers[1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frames
        cv2.imshow("Calibration", mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('c') and len(marker_centers) == 2:
            goalDetected = True
            print("Goal detected!")
    
    cv2.destroyAllWindows()

    return marker_centers

def predictBallTrajectory(cap, frame, pts, times, goal_positions):
    points = np.array(pts)
    timestamps = np.array(times)
    timestamps -= timestamps[-1]  # Normalize timestamps to start from 0

    # Fit a line to the x and y coordinates using linear regression
    A = np.vstack([timestamps, np.ones(len(timestamps))]).T  # Design matrix for linear regression
    velocity_x, _ = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]  # Slope for x-coordinates
    velocity_y, _ = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]  # Slope for y-coordinates

    # Predict future position after 500 ms
    prediction_time = 0.5
    pred_x = int(pts[0][0] + velocity_x * prediction_time)
    pred_y = int(pts[0][1] + velocity_y * prediction_time)

    cv2.line(frame, pts[0], (pred_x, pred_y), (255, 0, 0), 2)
    cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

    (x1, y1) = pts[0]
    (x2, y2) = (pred_x, pred_y)
    (x3, y3), (x4, y4) = goal_positions

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    if denom != 0:
        Px = int(((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom)
        Py = int(((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom)

        # Plot intersection point if between goal posts
        if (min(x3, x4) <= Px <= max(x3, x4)) and (min(y3, y4) <= Py <= max(y3, y4)):
            # Check if the ball is moving towards the goal
            goal_dx = x4 - x3
            goal_dy = y4 - y3

            # Perpendicular vector to the goal line
            perp_goal_dx = -goal_dy
            perp_goal_dy = goal_dx

            # Dot product to check direction
            dot_product = perp_goal_dx * velocity_x + perp_goal_dy * velocity_y

            if dot_product < 0:  # Change to track direction towards goal
                cv2.circle(frame, (Px, Py), 8, (0, 0, 255), -1)
                cv2.putText(frame, "Intercept", (Px + 10, Py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                
             # Calculate percentage of the goal line
            goal_length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            distance_to_marker1 = np.sqrt((Px - x3)**2 + (Py - y3)**2)
            percentage = distance_to_marker1 / goal_length

            # Display percentage on the frame
            percentage_text = f"Intercept: {percentage:.2f}"
            cv2.putText(frame, percentage_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        print("Waiting for camera connection...")
        time.sleep(3)
    print("Camera connected.")

    marker_centers = getGoalPositions(cap)

    pts = deque(maxlen=10)  # recent 10 positions
    times = deque(maxlen=10)
    createTrackbars()

    while True:
        if not cap.isOpened():
            print("Camera disconnected.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))
        sv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_range, upper_range = dynamicHSVCalibration()

        # Color mask to detect markers
        mask = cv2.inRange(sv, lower_range, upper_range)
        mask = morphCleaning(mask)

        # Mask out goalie marker regions, draw circles around them
        for i, center in enumerate(marker_centers):
            cv2.circle(mask, center, 10, 0, -1) # Mask out a circle around each marker
            cv2.circle(frame, center, 3, (0, 255, 255), -1)  # Draw small dot
            cv2.putText(frame, f"Marker {i+1}", (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.line(frame, marker_centers[0], marker_centers[1], (255, 0, 0), 2)

        # Find contour (ball)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 200:  # filter small objects/noise
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

                    # Draw detected puck
                    cv2.circle(frame, center, int(radius), (0,255,0), 2)
                    cv2.circle(frame, center, 4, (0,0,255), -1)

                    # Save position and timestamp
                    pts.appendleft(center)
                    times.appendleft(time.time())

        if len(pts) >= 10:
            predictBallTrajectory(cap, frame, pts, times, marker_centers)

        cv2.imshow("Ball Trajectory Prediction", frame)
        cv2.imshow("Calibration", mask)

        if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
            print("Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()