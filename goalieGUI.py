import PySimpleGUI as sg
import cv2
import numpy as np
from goalieModular import GoalieController

def init_window():
    """
    Phase 1: let user enter COM port & camera index, then
    click buttons to initialize serial, camera, capture.
    'Next' closes this window and returns a controller.
    """
    layout = [
        [ sg.Text('COM Port:'),  sg.Input('COM4', key='COM'),
          sg.Text('Camera #:'), sg.Input('1',   key='CAM') ],
        [ sg.Button('Init Serial'), sg.Text('', key='-SERIAL-') ],
        [ sg.Button('Init Camera'), sg.Text('', key='-CAMERA-') ],
        [ sg.Button('Start Capture'), sg.Text('', key='-CAP-') ],
        [ sg.Button('Calibrate Goals'), sg.Button('Exit') ]
    ]
    win = sg.Window('Initialization', layout, finalize=True)

    ctrl = None
    while True:
        event, vals = win.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            win.close()
            return None

        com  = vals['COM']
        cam  = int(vals['CAM'])

        if event == 'Init Serial':
            if ctrl is None:
                ctrl = GoalieController(com)
            try:
                ctrl.init_serial()
                win['-SERIAL-'].update('✔')
            except Exception:
                win['-SERIAL-'].update('✘')

        elif event == 'Init Camera' and ctrl:
            try:
                ctrl.cam_index = cam
                ctrl.init_camera()
                win['-CAMERA-'].update('✔')
            except Exception:
                win['-CAMERA-'].update('✘')

        elif event == 'Start Capture' and ctrl:
            ctrl.start_capture()
            win['-CAP-'].update('✔')

        elif event == 'Calibrate Goals' and ctrl:
            win.close()
            return ctrl

    # unreachable

def calibrate_window(ctrl):
    """
    Phase 2: pop up calibration window, driven by
    ctrl.calibrate_goals(window).  Returns goal_positions.
    """
    cal_layout = [
        [ sg.Image(key='-FRAME-'), sg.Image(key='-MASK-') ],
        [
          sg.Text('L-H'), sg.Slider((0,179),0,1,orientation='h', key='L_H'),
          sg.Text('L-S'), sg.Slider((0,255),0,1,orientation='h', key='L_S'),
          sg.Text('L-V'), sg.Slider((0,255),0,1,orientation='h', key='L_V')
        ],
        [
          sg.Text('U-H'), sg.Slider((0,179),179,1,orientation='h', key='U_H'),
          sg.Text('U-S'), sg.Slider((0,255),255,1,orientation='h', key='U_S'),
          sg.Text('U-V'), sg.Slider((0,255),255,1,orientation='h', key='U_V')
        ],
        [ sg.Button('Confirm'), sg.Button('Cancel') ],
        [ sg.Text('', key='-CAL-STATUS-') ]
    ]
    cal_win = sg.Window('Goal Calibration',
                        cal_layout,
                        finalize=True,
                        resizable=True)
    # hand off to your class method
    centers = ctrl.calibrate_goals(cal_win)
    cal_win.close()
    return centers

def track_window(ctrl):
    """
    Phase 3: ball tracking window.
    Shows live feed + mask with sliders.  Has a 'Lock' button
    to freeze HSV values for speed.
    """
    layout = [
      [ sg.Image(key='-TRACK-FRAME-'), sg.Image(key='-TRACK-MASK-') ],
      [ sg.Text('L-H'), sg.Slider((0,179),0,1,orientation='h', key='L_H'),
        sg.Text('L-S'), sg.Slider((0,255),0,1,orientation='h', key='L_S'),
        sg.Text('L-V'), sg.Slider((0,255),0,1,orientation='h', key='L_V') ],
      [ sg.Text('U-H'), sg.Slider((0,179),179,1,orientation='h', key='U_H'),
        sg.Text('U-S'), sg.Slider((0,255),255,1,orientation='h', key='U_S'),
        sg.Text('U-V'), sg.Slider((0,255),255,1,orientation='h', key='U_V') ],
      [ sg.Button('Lock HSV'), sg.Button('Reset Shot'), sg.Button('Exit Tracking') ],
      [ sg.Text('', key='-TRACK-STATUS-') ]
    ]
    win = sg.Window('Ball Tracking',
                    layout,
                    finalize=True,
                    resizable=True)

    locked = False
    hsv_bounds = None
    reset_shot = True

    while True:
        event, vals = win.read(timeout=20)
        if event in (sg.WIN_CLOSED, 'Exit Tracking'):
            break

        if event == 'Reset Shot':
            # clear the trajectory/history so we can collect a fresh shot
            ctrl.pts.clear()
            ctrl.times.clear()
            ctrl.last_sent_percentage = None
            ctrl.command_sent = False
            reset_shot = True
            win['-TRACK-STATUS-'].update('Ready for next shot ✔')
            continue

        ok, frame = ctrl.get_frame()
        if not ok or frame is None:
            continue

        # if not yet locked, read sliders each iteration
        if event == 'Lock HSV':
            hsv_bounds = (
                vals['L_H'], vals['L_S'], vals['L_V'],
                vals['U_H'], vals['U_S'], vals['U_V']
            )
            locked = True
            win['-TRACK-STATUS-'].update('HSV Locked ✔')

        if not locked:
            low_h, low_s, low_v = vals['L_H'], vals['L_S'], vals['L_V']
            up_h,  up_s,  up_v  = vals['U_H'], vals['U_S'], vals['U_V']
        else:
            low_h, low_s, low_v, up_h, up_s, up_v = hsv_bounds

        # TODO: overlay any tracking/prediction results here
        frame, precentage = ctrl.predict_ball_trajectory(win)

        small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2),
                           interpolation=cv2.INTER_AREA)
        hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        low  = np.array([low_h, low_s, low_v])
        up   = np.array([up_h, up_s, up_v])
        mask = cv2.inRange(hsv, low, up)

        if precentage is not None and reset_shot:
            cmd = ctrl.calculate_motor_command(precentage)
            ctrl.send_motor_command(cmd)
            print(f"Motor Command: {cmd}")
            reset_shot = False

        # update images
        win['-TRACK-FRAME-'].update(data=cv2.imencode('.png', small)[1].tobytes())
        win['-TRACK-MASK-'].update(data=cv2.imencode('.png', mask)[1].tobytes())

    win.close()

def main():
    # Phase 1: Initialization
    ctrl = init_window()
    if not ctrl:
        return

    # Phase 2: Calibration
    centers = calibrate_window(ctrl)
    if not centers:
        return

    # Phase 3: Tracking
    track_window(ctrl)


if __name__ == '__main__':
    main()
