import cv2, time, serial, threading, queue
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import PySimpleGUI as sg
import serial.tools.list_ports

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

        self.hsv_locked = False
        self.hsv_bounds = None

    def init_serial(self):
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                print(port.device)
            for port in ports:
                try:
                    temp_port = serial.Serial(port.device)
                    if temp_port.is_open:
                        temp_port.close()
                except Exception as e:
                    print(f"Error closing port {port.device}: {e}")
            # Initialize serial communication with the motor controller
            self.ser = serial.Serial(self.comPort, 115200, timeout=1)
            time.sleep(2)
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
            return True, frame
        except queue.Empty:
            return False, None

    def calibrate_goals(self, window):
        confirm_button = window['Confirm']
        while True:
            event, values = window.read(timeout=20)
            if event in (sg.WIN_CLOSED, 'Cancel', 'Exit'):
                sg.popup("Calibration cancelled.")
                return None
            self.goal_positions = []

            low_h, low_s, low_v = values['L_H'], values['L_S'], values['L_V']
            up_h,  up_s,  up_v  = values['U_H'], values['U_S'], values['U_V']
            lower_range = np.array([low_h, low_s, low_v])
            upper_range = np.array([up_h, up_s, up_v])

            ok, frame = self.get_frame()

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
                self.goal_positions[0], self.goal_positions[1] = self.goal_positions[1], self.goal_positions[0]
                cv2.line(frame, self.goal_positions[0], self.goal_positions[1], (255,0,0), 2)

            # 5. Push back to GUI
            frame_bytes = cv2.imencode('.png', frame)[1].tobytes()
            mask_bytes  = cv2.imencode('.png', mask)[1].tobytes()
            window['-FRAME-'].update(data=frame_bytes)
            window['-MASK-'].update(data=mask_bytes)

            if len(self.goal_positions) == 2:
                confirm_button.update(disabled=False)
            else:
                confirm_button.update(disabled=True)
            
            # if they hit Confirm and we have two markers, we’re done
            if event == 'Confirm':
                if len(self.goal_positions) == 2:
                    return self.goal_positions
                else:
                    sg.popup("Two markers not found. Adjust sliders and try again.")
    
    def predict_ball_trajectory(self, window):
        ok, frame = self.get_frame()
        if not ok or frame is None:
            return None, None, None
        frame = cv2.resize(frame, (540, 960))
        event, values = window.read(timeout=20)
        if self.hsv_locked and self.hsv_bounds is not None:
            low_h, low_s, low_v, up_h, up_s, up_v = self.hsv_bounds
        else:
            # assume you stored the most recent slider values on the controller
            low_h, low_s, low_v = (values['L_H'],
                                values['L_S'],
                                values['L_V'])
            up_h,  up_s,  up_v  = (values['U_H'],
                                values['U_S'],
                                values['U_V'])

        # 2) Build and apply the mask
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([low_h, low_s, low_v])
        upper = np.array([up_h,  up_s,  up_v])
        mask = cv2.inRange(hsv, lower, upper)
        for i, center in enumerate(self.goal_positions):
            cv2.circle(mask, center, 70, 0, -1)
            cv2.circle(mask, (500, 265), 30, 0, -1)
        cv2.line(mask, self.goal_positions[0], self.goal_positions[1], 0, 200)

        # Find contour (ball) in the cropped mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if contours:
            # Sort contours by area and process the largest one
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            # Check if the contour is circular and within the frame
            if 100 < area < 1e5:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                x, y, w, h = cv2.boundingRect(c)
                if x > 0 and y > 0 and (x + w) < frame.shape[1] and (y + h) < frame.shape[0]:  # Fully within frame
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                        # Draw detected ball
                        cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

                        self.pts.appendleft(center)
                        self.times.appendleft(time.time())

        for (cx, cy) in self.goal_positions:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if len(self.pts) >= 10:
            points = np.array(self.pts)
            timestamps = np.array(self.times)
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
            pred_x = int(self.pts[0][0] + velocity_x * prediction_time)
            pred_y = int(self.pts[0][1] + velocity_y * prediction_time)

            cv2.line(frame, self.pts[0], (pred_x, pred_y), (255, 0, 0), 2)
            # cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

            (x1, y1) = self.pts[0]
            (x2, y2) = (pred_x, pred_y)
            (x3, y3), (x4, y4) = self.goal_positions

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
                    # dot_product = perp_goal_dx * velocity_x + perp_goal_dy * velocity_y

                    # Calculate percentage of the goal line
                    goal_length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                    distance_to_marker1 = np.sqrt((Px - x3)**2 + (Py - y3)**2)
                    percentage = distance_to_marker1 / goal_length

                    return frame, mask, percentage
        return frame, mask, None
    
    def calculate_motor_command(self, percentage):
        return int((1950 - 80) * percentage)  # Return as integer

    def send_motor_command(self, command):
        package = str(command).encode()
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