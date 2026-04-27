import cv2
import time
import board
import busio
import threading
import signal
import sys
from flask import Flask, Response, render_template_string, jsonify
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'best_rknn_model'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

# FLASK CONFIG
HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# --- TUNING ---
ROI_VERTICAL_CUTOFF = 0.7
Kp = 0.0007
Kd = 0.0005
BASE_SPEED = 0.12
LANE_WIDTH_PIXELS = 450

# STOP SIGN LOGIC
STOP_DURATION = 2.0
STOP_COOLDOWN = 5.0
STOP_THRESHOLD_Y = CAMERA_HEIGHT * 0.8

# MOTOR PHYSICS
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.8

# --- GLOBAL STATE ---
output_frame = None
lock = threading.Lock()

# robot_enabled: False = hold position (motors idle), True = running
robot_enabled = False
robot_enabled_lock = threading.Lock()

# Shared reference to motor stop function so Ctrl+C can call it
_stop_all_fn = None

# --- FLASK APP ---
app = Flask(__name__)

# --- Motor Class ---
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


# --- ROBOT LOGIC THREAD ---
def robot_control_loop():
    global output_frame, lock, robot_enabled, _stop_all_fn

    # 1. Init Hardware
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100
        left_motors  = [Motor(pca, 0, 1), Motor(pca, 6, 7)]
        right_motors = [Motor(pca, 2, 3), Motor(pca, 4, 5)]
    except Exception as e:
        print(f"Hardware Init Error: {e}")
        return

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)
        left  = fwd + steer
        right = fwd - steer
        max_val = max(abs(left), abs(right))
        if max_val > 1.0:
            left  /= max_val
            right /= max_val
        for m in left_motors:  m.set_speed(left)
        for m in right_motors: m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors:
            m.stop()

    # Expose stop_all so the SIGINT handler can reach it
    _stop_all_fn = stop_all

    # 2. Load Model
    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)

    prev_error = 0
    last_stop_time = 0

    print("\n--- ROBOT READY — press Start in the web UI to begin ---")

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError("Cannot open camera (VideoCapture(0) failed)")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Flip BEFORE YOLO inference (fix mirrored webcam)
            frame = cv2.flip(frame, 1)

            # -------------------------------------------------------
            # GATE: if robot not enabled, show live feed but don't drive
            # -------------------------------------------------------
            with robot_enabled_lock:
                enabled = robot_enabled

            if not enabled:
                # Still stream video so the operator can see the camera
                waiting_frame = frame.copy()
                cv2.putText(waiting_frame, "WAITING FOR START",
                            (80, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                with lock:
                    output_frame = waiting_frame.copy()
                stop_all()   # ensure motors are off while waiting
                time.sleep(0.03)
                continue

            # Run inference on the flipped frame
            results = model.predict(source=frame, conf=0.4, imgsz=640, verbose=False)
            result  = results[0]
            boxes   = result.boxes

            # --- VISION PROCESSING ---
            best_y_x = None
            best_w_x = None
            max_y_area = 0
            max_w_area = 0
            stop_requested = False

            current_time = time.time()

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                # Red Line Check
                if cls == 'redline':
                    if y > STOP_THRESHOLD_Y:
                        if (current_time - last_stop_time) > STOP_COOLDOWN:
                            stop_requested = True

                # Lane Check (Turn Later Logic)
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

            # --- VIDEO FRAME UPDATE ---
            annotated_frame = result.plot()

            # --- CONTROL LOGIC ---

            # 1. Stop request (commented-out stop logic preserved)
            if stop_requested:
                continue

            # 2. Calculate Target
            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2
            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH_PIXELS / 2)
            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH_PIXELS / 2)
            else:
                target_x = CENTER_X

            # 3. PD steering
            error      = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error
            steering   = (error * Kp) + (derivative * Kd)

            # --- DEBUG TEXT OVERLAY ---
            debug_lines = [
                f"best_w_x: {best_w_x}",
                f"best_y_x: {best_y_x}",
                f"target_x: {target_x:.1f}",
                f"CENTER_X: {CENTER_X:.1f}",
                f"error: {error:.1f}",
                f"steering: {steering:.4f}",
                f"BASE_SPEED: {BASE_SPEED:.2f}",
            ]
            x0, y0 = 10, 25
            for i, line in enumerate(debug_lines):
                ypos = y0 + i * 22
                cv2.putText(annotated_frame, line, (x0, ypos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, line, (x0, ypos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            set_drive(BASE_SPEED, steering)

            # --- VISUAL DEBUGGING ON VIDEO ---
            debug_y = int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF) + 20
            cv2.circle(annotated_frame, (int(target_x), debug_y), 10, (0, 255, 0), -1)
            cv2.line(annotated_frame,
                     (int(CENTER_X), 0), (int(CENTER_X), CAMERA_HEIGHT),
                     (255, 255, 255), 1)

            with lock:
                output_frame = annotated_frame.copy()

    except Exception as e:
        print(f"Robot Loop Error: {e}")
    finally:
        stop_all()
        print("Robot Loop Ended — motors stopped.")


# --- FLASK STREAMING ---

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.03)
                continue
            flag, encodedImage = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
            frame_bytes = bytearray(encodedImage)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Robot Vision</title>
        <style>
            body {
                background: #111;
                color: #eee;
                text-align: center;
                font-family: monospace;
                user-select: none;
            }
            h1 { margin-bottom: 4px; }
            #status {
                font-size: 1.1em;
                margin: 8px 0 12px;
                letter-spacing: 1px;
            }
            .btn {
                display: inline-block;
                padding: 12px 36px;
                margin: 0 8px 16px;
                font-size: 1.1em;
                font-family: monospace;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                transition: opacity 0.15s;
            }
            .btn:disabled { opacity: 0.35; cursor: default; }
            #btn-start { background: #2a9d2a; color: #fff; }
            #btn-stop  { background: #c0392b; color: #fff; }
            img { border: 2px solid #555; display: block; margin: 0 auto; }
        </style>
    </head>
    <body>
        <h1>RADXA ROBOT V11</h1>
        <div id="status">● IDLE — motors off</div>

        <button class="btn" id="btn-start" onclick="setRobot(true)">▶ START</button>
        <button class="btn" id="btn-stop"  onclick="setRobot(false)" disabled>■ STOP</button>

        <br>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">

        <script>
            function setRobot(enable) {
                fetch(enable ? '/start' : '/stop', { method: 'POST' })
                    .then(r => r.json())
                    .then(d => {
                        document.getElementById('status').textContent =
                            d.enabled
                                ? '● RUNNING — robot active'
                                : '● IDLE — motors off';
                        document.getElementById('btn-start').disabled =  d.enabled;
                        document.getElementById('btn-stop').disabled  = !d.enabled;
                    });
            }
        </script>
    </body>
    </html>
    """)


@app.route('/start', methods=['POST'])
def start_robot():
    global robot_enabled
    with robot_enabled_lock:
        robot_enabled = True
    print(">>> Robot STARTED via web UI")
    return jsonify({"enabled": True})


@app.route('/stop', methods=['POST'])
def stop_robot():
    global robot_enabled
    with robot_enabled_lock:
        robot_enabled = False
    print(">>> Robot STOPPED via web UI")
    return jsonify({"enabled": False})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- SIGINT HANDLER (Ctrl+C) ---
def handle_sigint(sig, frame):
    print("\nCtrl+C detected — stopping motors and exiting.")
    # Set flag first so the control loop stops driving
    with robot_enabled_lock:
        global robot_enabled
        robot_enabled = False
    # Directly call hardware stop if available
    if _stop_all_fn is not None:
        try:
            _stop_all_fn()
        except Exception:
            pass
    sys.exit(0)


# --- MAIN ---
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)

    t = threading.Thread(target=robot_control_loop, daemon=True)
    t.start()

    print(f"Starting Web Server at http://{HOST_IP}:{HOST_PORT}")
    # use_reloader=False is required — reloader forks and breaks signal handling
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True, use_reloader=False)
