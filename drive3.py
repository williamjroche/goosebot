import cv2
import time
import board
import busio
import threading
from flask import Flask, Response, render_template_string
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = 'best_rknn_model'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# --- TUNING ---
ROI_VERTICAL_CUTOFF = 0.67
Kp = 0.0007
Kd = 0.00054
BASE_SPEED = 0.15
LANE_WIDTH_PIXELS = 450 

MIN_MOTOR_POWER = 0.07  
MAX_STEER = 0.8

# --- GLOBAL STATE ---
output_frame = None
lock = threading.Lock()

# ✅ THREAD SAFE EMERGENCY STOP SYSTEM
state_lock = threading.Lock()
emergency_stop = False

app = Flask(__name__)

# --- MOTOR CLASS ---
class Motor:
    def __init__(self, pca, in1, in2):
        self.pca = pca
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            pwm = 0
        else:
            abs_s = abs(speed)
            mapped_speed = MIN_MOTOR_POWER + (abs_s * (1.0 - MIN_MOTOR_POWER))
            pwm = int(min(mapped_speed, 1.0) * 65535)

        if speed > 0:
            self.in1.duty_cycle = pwm
            self.in2.duty_cycle = 0
        elif speed < 0:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm
        else:
            self.stop()

    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0


# --- FLASK CONTROL ENDPOINTS ---
@app.route('/stop')
def stop_robot():
    global emergency_stop
    with state_lock:
        emergency_stop = True
    print("🛑 EMERGENCY STOP TRIGGERED")
    return "STOPPED"


@app.route('/start')
def start_robot():
    global emergency_stop
    with state_lock:
        emergency_stop = False
    print("🟢 ROBOT STARTED")
    return "STARTED"


# --- ROBOT LOOP ---
def robot_control_loop():
    global output_frame, lock

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100

        left_motors = [Motor(pca, 0, 1), Motor(pca, 6, 7)]
        right_motors = [Motor(pca, 2, 3), Motor(pca, 4, 5)]

    except Exception as e:
        print(f"Hardware Init Error: {e}")
        return

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)

        left = fwd + steer
        right = fwd - steer

        max_val = max(abs(left), abs(right))
        if max_val > 1.0:
            left /= max_val
            right /= max_val

        for m in left_motors:
            m.set_speed(left)
        for m in right_motors:
            m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors:
            m.stop()

    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    for _ in range(5):
        cap.read()

    prev_error = 0

    input("\n--- READY. Press Enter to start ---")
    print("--- ROBOT STARTED ---")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            # ✅ SAFE SNAPSHOT OF STATE (prevents race condition)
            with state_lock:
                stop_flag = emergency_stop

            if stop_flag:
                stop_all()
                time.sleep(0.05)
                continue

            results = model.predict(source=frame, conf=0.40, imgsz=640, verbose=False)
            result = results[0]
            boxes = result.boxes

            best_y_x = None
            best_w_x = None
            max_y_area = 0
            max_w_area = 0

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                cutoff_pixel = CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF
                if y < cutoff_pixel:
                    continue

                area = w * h

                if cls == 'yellowline' and area > max_y_area:
                    max_y_area = area
                    best_y_x = x
                elif cls == 'whiteline' and area > max_w_area:
                    max_w_area = area
                    best_w_x = x

            annotated_frame = result.plot()

            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2
            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH_PIXELS / 2)
            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH_PIXELS / 2)
            else:
                target_x = CENTER_X

            error = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error

            steering = (error * Kp) + (derivative * Kd)

            set_drive(BASE_SPEED, steering)

            with lock:
                output_frame = annotated_frame.copy()

    except Exception as e:
        print(f"Robot Loop Error: {e}")
    finally:
        stop_all()
        print("Robot Loop Ended")


# --- STREAM ---
def generate_frames():
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string("""
    <html>
    <body style="background:#111;color:white;text-align:center;">
        <h2>Robot Vision</h2>
        <img src="/video_feed" width="640">
        <p>/stop and /start control robot state</p>
    </body>
    </html>
    """)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- MAIN ---
if __name__ == "__main__":

    threading.Thread(target=robot_control_loop, daemon=True).start()

    print("Server running at http://0.0.0.0:5000")
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True, use_reloader=False)
