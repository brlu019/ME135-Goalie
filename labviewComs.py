import PySimpleGUI as sg
import cv2

layout = [
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Text("Low H"), sg.Slider((0,179), 0, 1, orientation='h', key='L_H')],
    [sg.Button("Calibrate"), sg.Button("Start"), sg.Button("Exit")]
]

window = sg.Window("Goalie CV", layout, location=(100,100))

cap = cv2.VideoCapture(0)
while True:
    event, values = window.read(timeout=20)
    if event in ("Exit", sg.WIN_CLOSED):
        break

    ret, frame = cap.read()
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window["-IMAGE-"].update(data=imgbytes)

    if event == "Calibrate":
        low_h = int(values['L_H'])
        # … your calibration logic
    if event == "Start":
        print("Starting...")

cap.release()
window.close()
