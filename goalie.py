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
                cv2.putText(frame, f"{marker_centers[0]}", (marker_centers[0][0]+10, marker_centers[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"{marker_centers[1]}", (marker_centers[1][0]+10, marker_centers[1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frames
        cv2.imshow("Calibration", mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('c') and len(marker_centers) == 2:
            goalDetected = True
            print("Goal detected!")
    
    cv2.destroyAllWindows()

    return marker_centers

def predictBallTrajectory(cap, frame, pts, times):
    dx = pts[0][0] - pts[-1][0]
    dy = pts[0][1] - pts[-1][1]
    dt = times[0] - times[-1]

    if dt > 0:
        velocity_x = dx / dt
        velocity_y = dy / dt

        # Predict future position (e.g., after 500 ms)
        prediction_time = 0.5
        pred_x = int(pts[0][0] + velocity_x * prediction_time)
        pred_y = int(pts[0][1] + velocity_y * prediction_time)

        cv2.line(frame, pts[0], (pred_x, pred_y), (255, 0, 0), 2)
        cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

    return velocity_x, velocity_y

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

        # Mask out goalie marker regions
        for center in marker_centers:
            cv2.circle(mask, center, 20, 0, -1)  # Mask out a circle around each marker

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

        if len(pts) >= 2:
            predictBallTrajectory(cap, frame, pts, times)

        cv2.imshow("Ball Trajectory Prediction", frame)
        cv2.imshow("Calibration", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()