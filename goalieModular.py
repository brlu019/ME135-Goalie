import cv2, time, serial, threading, queue
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import PySimpleGUI as sg

class GoalieController:
    def __init__(self, comPort):
        self.comPort = comPort
        self.ser = None
        self.cap = None

        self.pts = deque(maxlen=10)  # recent 10 positions
        self.times = deque(maxlen=10)
        self.goal_positions = []
        self.percentage = None
        self.last_sent_percentage = None
        self.command_sent = True  # Flag to track if a command has been sent, initially true

        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def init_serial(self):
        try:
            # Initialize serial communication with the motor controller
            self.ser = serial.Serial(self.comPort, 115200, timeout=1)
            time.sleep(2)
            self.ser.open()
            print("Serial port opened successfully.")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            time.sleep(3)

    def init_camera(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(1)
        while not self.cap.isOpened():
            print("Waiting for camera connection...")
            time.sleep(3)
        print("Camera connected.")

    def start_capture(self):
        def capture():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
        self.capture_thread = threading.Thread(target=capture, daemon=True)
        self.capture_thread.start()

    def get_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

    def calibrate_goals(self, low_h, low_s, low_v, up_h, up_s, up_v, window):
        lower_range = np.array([low_h, low_s, low_v])
        upper_range = np.array([up_h, up_s, up_v])

        while True:
            ret, frame = self.get_frame()
            if not ret:
                continue
            frame = cv2.resize(frame, (540, 960))
            sv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(sv, lower_range, upper_range)
            cv2.circle(mask, (500, 265), 30, 0, -1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >= 2:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    peri = cv2.arcLength(cnt, True)
                    circ = 4*np.pi*(area/(peri*peri)) if peri>0 else 0
                    if 100 < area < 1e5 and circ > 0.8:
                        M = cv2.moments(cnt)
                        if M["m00"]!=0:
                            cx = int(M["m10"]/M["m00"])
                            cy = int(M["m01"]/M["m00"])
                            self.goal_positions.append((cx, cy))

            # 4. Draw results on frame
            for (cx,cy) in self.goal_positions:
                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            if len(self.goal_positions)==2:
                cv2.line(frame, self.goal_positions[0], self.goal_positions[1], (255,0,0), 2)

            # 5. Push back to GUI
            frame_bytes = cv2.imencode('.png', frame)[1].tobytes()
            mask_bytes  = cv2.imencode('.png', mask)[1].tobytes()
            window['-FRAME-'].update(data=frame_bytes)
            window['-MASK-'].update(data=mask_bytes)

            # 6. GUI event loop
            event, _ = window.read(timeout=20)
            if event in (sg.WIN_CLOSED, 'Exit'):
                sg.popup("Calibration cancelled.")
                return None
            if event == 'Confirm':
                if len(self.goal_positions)==2:
                    return self.goal_positions
                else:
                    sg.popup("Two markers not found. Adjust sliders and try again.")
    
    def calculateMotorCommand(self, percentage):
        return int((1950 - 80) * percentage)  # Return as integer

    def sendMotorCommand(self, command):
        package = command.encode()
        self.ser.reset_input_buffer()  # Clear input buffer before sending command
        self.ser.reset_output_buffer()  # Clear output buffer before sending command
        try:
            self.ser.write(package)  # Send command to motor controller
            print(f"Motor command sent: {command}")
        except serial.SerialException as e:
            # print(f"Error sending command: {e}")
            return
        # response = ser.readline().decode().strip()
        # if response:
        #     print(f"Response from motor controller: {response}")
        # else:
        #     print("No response from motor controller.")





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

def getGoalPositions(cap):
    goalDetected = False
    createTrackbars()

    print("Press 'c' to confirm goal positions.")

    while not goalDetected:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (540, 960))
        sv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_range, upper_range = dynamicHSVCalibration()

        # Color mask to detect markers
        mask = cv2.inRange(sv, lower_range, upper_range)
        # mask = morphCleaning(mask)
        cv2.circle(mask, (500, 265), 30, 0, -1)

        # Find contours (markers)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        goal_positions = []

        if len(contours) >= 5:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Get the two largest contours

            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0

                # Check if the contour is circular and within the frame
                if 100 < area and circularity > 0.8:  # Circularity threshold (1.0 is a perfect circle)
                    x, y, w, h = cv2.boundingRect(cnt)
                    if x > 0 and y > 0 and (x + w) < frame.shape[1] and (y + h) < frame.shape[0]:  # Fully within frame
                        M = cv2.moments(cnt)  # Calculate marker centers
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            goal_positions.append((cx, cy))
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Draw detected marker centers

            if len(goal_positions) == 2:
                # Sort markers by x-coordinate (leftmost first)
                goal_positions = sorted(goal_positions, key=lambda center: center[0])

                # Draw a line between the two markers
                cv2.line(frame, goal_positions[0], goal_positions[1], (255, 0, 0), 2)

                # Label the markers
                cv2.putText(frame, f"Marker 1 {goal_positions[0]}", 
                            (goal_positions[0][0] + 10, goal_positions[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Marker 2 {goal_positions[1]}", 
                            (goal_positions[1][0] + 10, goal_positions[1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frames
        cv2.imshow("Calibration", mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            if len(goal_positions) == 2:
                goalDetected = True
                print("Goal detected!")
            else:
                print("Please ensure two markers are visible.")
    
    cv2.destroyAllWindows()

    return goal_positions

def predictBallTrajectory(cap, frame, pts, times, goal_positions):
    points = np.array(pts)
    timestamps = np.array(times)
    timestamps -= timestamps[-1]  # Normalize timestamps to start from 0

    if np.abs(timestamps[0] - timestamps[-1]) < 1e-3:
        print("Not enough time difference between points.")
        return  # Avoid unstable regression if too little time difference

    # Fit a line to the x and y coordinates using linear regression
    A = np.vstack([timestamps, np.ones(len(timestamps))]).T  # Design matrix for linear regression
    velocity_x, _ = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]  # Slope for x-coordinates
    velocity_y, _ = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]  # Slope for y-coordinates

    # Predict future position after 200 ms
    prediction_time = 0.2
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

            # Calculate percentage of the goal line
            goal_length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            distance_to_marker1 = np.sqrt((Px - x3)**2 + (Py - y3)**2)
            percentage = distance_to_marker1 / goal_length

            return percentage
    return None

def process_frame(frame, goal_positions):
    frame = cv2.resize(frame, (540, 960))
    sv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range, upper_range = dynamicHSVCalibration()
    mask = cv2.inRange(sv, lower_range, upper_range)
    # mask = morphCleaning(mask)

    # Draw the markers, etc.
    for i, center in enumerate(goal_positions):
        cv2.circle(mask, center, 70, 0, -1)
        cv2.circle(mask, (500, 265), 30, 0, -1)
        cv2.circle(frame, center, 3, (0, 255, 255), -1)
    cv2.line(frame, goal_positions[0], goal_positions[1], (255, 0, 0), 2)

    crop_y = min(goal_positions[0][1], goal_positions[1][1])
    # crop_x_min = goal_positions[0][0]
    # crop_x_max = goal_positions[1][0]
    cropped_frame = frame[crop_y:, :]
    cropped_mask = mask[crop_y:, :]
    
    # Return processed ROI for further processing
    return cropped_frame, cropped_mask

def tracking():
    print("Press 'r' to reset for shot.")
    while True:
        if not cap.isOpened():
            print("Camera disconnected.")
            break

        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        future = executor.submit(process_frame, frame, goal_positions)
        cropped_frame, cropped_mask = future.result()

        # Find contour (ball) in the cropped mask
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if contours:
            # Sort contours by area and process the largest one
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            # Check if the contour is circular and within the frame
            if 100 < area < 7500:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                x, y, w, h = cv2.boundingRect(c)
                if x > 0 and y > 0 and (x + w) < frame.shape[1] and (y + h) < frame.shape[0]:  # Fully within frame
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                        # Draw detected ball
                        cv2.circle(cropped_frame, center, int(radius), (0, 255, 0), 2)

                        pts.appendleft(center)
                        times.appendleft(time.time())

        if len(pts) >= 10 and not command_sent:  # Only process if no command has been sent
            percentage = predictBallTrajectory(cap, cropped_frame, pts, times, goal_positions)
            if percentage is not None:
                if last_sent_percentage is None or abs(percentage - last_sent_percentage) > 0.03:
                    # Calculate motor command based on percentage
                    command = calculateMotorCommand(percentage)
                    sendMotorCommand(str(command))
                    last_sent_percentage = percentage
                    command_sent = True  # Set flag to indicate command has been sent
                    print("Motor command sent. Waiting for reset...")

        # Reset logic
        if cv2.waitKey(1) & 0xFF == ord('r'):  # Reset when 'r' is pressed
            command_sent = False
            pts.clear()
            times.clear()
            last_sent_percentage = None
            print("Reset complete. Ready for next shot.")
            time.sleep(1)

        cv2.imshow("Ball Trajectory Prediction", cropped_frame)
        cv2.imshow("Calibration", cropped_mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# def main():
    # # Make sure serial port is available
    # while not ser.is_open:
    #     print("Waiting for serial connection...")
    #     time.sleep(3)

    # # Initialize the camera
    # cap = cv2.VideoCapture(1)
    # while not cap.isOpened():
    #     print("Waiting for camera connection...")
    #     time.sleep(3)
    # print("Camera connected.")

    # Start frame capture thread
    # threading.Thread(target=frame_capture_thread, args=(cap, frame_queue), daemon=True).start()

    # goal_positions = getGoalPositions(cap)

    # pts = deque(maxlen=10)  # recent 6 positions
    # times = deque(maxlen=10)
    # createTrackbars()

    # percentage = None
    # last_sent_percentage = None
    # command_sent = True  # Flag to track if a command has been sent, initally true
    # print("Press 'r' to reset for shot.")

    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # while True:
    #     if not cap.isOpened():
    #         print("Camera disconnected.")
    #         break

    #     if frame_queue.empty():
    #         continue

    #     frame = frame_queue.get()
    #     future = executor.submit(process_frame, frame, goal_positions)
    #     cropped_frame, cropped_mask = future.result()

    #     # Find contour (ball) in the cropped mask
    #     contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     center = None

    #     if contours:
    #         # Sort contours by area and process the largest one
    #         c = max(contours, key=cv2.contourArea)
    #         area = cv2.contourArea(c)

    #         # Check if the contour is circular and within the frame
    #         if 100 < area < 7500:
    #             ((x, y), radius) = cv2.minEnclosingCircle(c)
    #             x, y, w, h = cv2.boundingRect(c)
    #             if x > 0 and y > 0 and (x + w) < frame.shape[1] and (y + h) < frame.shape[0]:  # Fully within frame
    #                 M = cv2.moments(c)
    #                 if M["m00"] != 0:
    #                     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    #                     # Draw detected ball
    #                     cv2.circle(cropped_frame, center, int(radius), (0, 255, 0), 2)

    #                     pts.appendleft(center)
    #                     times.appendleft(time.time())

    #     if len(pts) >= 10 and not command_sent:  # Only process if no command has been sent
    #         percentage = predictBallTrajectory(cap, cropped_frame, pts, times, goal_positions)
    #         if percentage is not None:
    #             if last_sent_percentage is None or abs(percentage - last_sent_percentage) > 0.03:
    #                 # Calculate motor command based on percentage
    #                 command = calculateMotorCommand(percentage)
    #                 sendMotorCommand(str(command))
    #                 last_sent_percentage = percentage
    #                 command_sent = True  # Set flag to indicate command has been sent
    #                 print("Motor command sent. Waiting for reset...")

    #     # Reset logic
    #     if cv2.waitKey(1) & 0xFF == ord('r'):  # Reset when 'r' is pressed
    #         command_sent = False
    #         pts.clear()
    #         times.clear()
    #         last_sent_percentage = None
    #         print("Reset complete. Ready for next shot.")
    #         time.sleep(1)

    #     cv2.imshow("Ball Trajectory Prediction", cropped_frame)
    #     cv2.imshow("Calibration", cropped_mask)

    #     if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
    #         print("Exiting...")
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()