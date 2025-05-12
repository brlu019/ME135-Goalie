import cv2, time, serial, threading, queue
import numpy as np
import PySimpleGUI as sg
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class GoalieController:
    def __init__(self, com_port, cam_index=1):
        self.com_port = com_port
        self.cam_index = cam_index
        self.ser = None
        self.cap = None

        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None

    def init_serial(self):
        try:
            self.ser = serial.Serial(self.com_port, 115200, timeout=1)
            time.sleep(2)
            sg.popup_auto_close(f"Serial on {self.com_port} opened!", auto_close_duration=1)
        except serial.SerialException as e:
            sg.popup_error(f"Error opening {self.com_port}: {e}")

    def init_camera(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            sg.popup_error(f"Cannot open camera {self.cam_index}")
        else:
            sg.popup_auto_close(f"Camera {self.cam_index} ready!", auto_close_duration=1)

    def start_capture(self):
        if self.cap is None or not self.cap.isOpened():
            sg.popup_error("Camera not initialized!")
            return

        def _capture():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                if not self.frame_queue.empty():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass
                self.frame_queue.put(frame)
        self.capture_thread = threading.Thread(target=_capture, daemon=True)
        self.capture_thread.start()
        sg.popup_auto_close("Capture thread started!", auto_close_duration=1)

    def get_frame(self):
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, None

    def calibrate_goals(self, window):
        """Blocks until Confirm with 2 markers, updates '-FRAME-' & '-MASK-' in window."""
        while True:
            ok, frame = self.get_frame()
            if not ok:
                event, _ = window.read(timeout=20)
                if event in ('Exit','Cancel'): return None
                continue

            # grab slider values
            vals = window.read(timeout=0)[1]
            low = np.array([vals['L_H'], vals['L_S'], vals['L_V']])
            up  = np.array([vals['U_H'], vals['U_S'], vals['U_V']])

            # process
            frame_resized = cv2.resize(frame, (540,960))
            hsv  = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, low, up)
            cv2.circle(mask, (500,265), 30, 0, -1)

            # find two circular contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                area = cv2.contourArea(cnt); peri = cv2.arcLength(cnt,True)
                circ = 4*np.pi*(area/(peri*peri)) if peri>0 else 0
                if 100<area<1e5 and circ>0.8:
                    M = cv2.moments(cnt)
                    if M['m00']!=0:
                        x = int(M['m10']/M['m00']); y = int(M['m01']/M['m00'])
                        centers.append((x,y))

            # draw
            for (x,y) in centers:
                cv2.circle(frame_resized,(x,y),5,(0,0,255),-1)
            if len(centers)==2:
                cv2.line(frame_resized, centers[0], centers[1], (255,0,0),2)

            # update GUI images
            window['-FRAME-'].update(data=cv2.imencode('.png', frame_resized)[1].tobytes())
            window['-MASK-'].update(data=cv2.imencode('.png', mask)[1].tobytes())

            # handle events
            event, _ = window.read(timeout=20)
            if event in ('Exit','Cancel'):
                return None
            if event=='Confirm':
                if len(centers)==2:
                    sg.popup_auto_close(f"Found: {centers}", auto_close_duration=1)
                    return centers
                else:
                    sg.popup("Need exactly two markers. Adjust sliders.")

def main():
    # 1) Build layout
    slider_row1 = [
      sg.Text('L-H'), sg.Slider((0,179),0,1,orientation='h', key='L_H'),
      sg.Text('L-S'), sg.Slider((0,255),0,1,orientation='h', key='L_S'),
      sg.Text('L-V'), sg.Slider((0,255),0,1,orientation='h', key='L_V')
    ]
    slider_row2 = [
      sg.Text('U-H'), sg.Slider((0,179),179,1,orientation='h', key='U_H'),
      sg.Text('U-S'), sg.Slider((0,255),255,1,orientation='h', key='U_S'),
      sg.Text('U-V'), sg.Slider((0,255),255,1,orientation='h', key='U_V')
    ]
    button_row = [
      sg.Button('Init Serial'), sg.Button('Init Camera'),
      sg.Button('Start Capture'), sg.Button('Calibrate Goals'),
      sg.Button('Exit')
    ]
    img_row = [[sg.Image(key='-FRAME-'), sg.Image(key='-MASK-')]]

    layout = img_row + [slider_row1, slider_row2, button_row]
    window = sg.Window('Goalie Control', layout, finalize=True)

    # 2) Create controller
    ctrl = GoalieController(com_port='COM4')

    # 3) Event loop
    while True:
        event, values = window.read(timeout=30)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Init Serial':
            ctrl.init_serial()
        elif event == 'Init Camera':
            ctrl.init_camera()
        elif event == 'Start Capture':
            ctrl.start_capture()
        elif event == 'Calibrate Goals':
            centers = ctrl.calibrate_goals(window)
            if centers:
                window['-GOALS-'] = sg.Text(f"Markers: {centers}")
        # continuously display live feed if capturing
        ok, frame = ctrl.get_frame()
        if ok:
            window['-FRAME-'].update(data=cv2.imencode('.png', frame)[1].tobytes())

    window.close()
    if ctrl.cap: ctrl.cap.release()

if __name__ == '__main__':
    main()
