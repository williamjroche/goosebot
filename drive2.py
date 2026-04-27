import cv2
import time
import board
import busio
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'best_rknn_model'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

# --- TUNING ---
ROI_VERTICAL_CUTOFF = 0.7
Kp = 0.0007
Kd = 0.0005
BASE_SPEED = 0.12
LANE_WIDTH_PIXELS = 450

# STOP SIGN LOGIC
STOP_COOLDOWN = 5.0
STOP_THRESHOLD_Y = CAMERA_HEIGHT * 0.8

# MOTOR PHYSICS
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.8


# --- Motor Class ---
class Motor:
    def __init__(self, pca, in1, in2):
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            self.stop()
            return
        mapped = MIN_MOTOR_POWER + (abs(speed) * (1.0 - MIN_MOTOR_POWER))
        pwm = int(min(mapped, 1.0) * 65535)
        if speed > 0:
            self.in1.duty_cycle = pwm
            self.in2.duty_cycle = 0
        else:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm

    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0


# --- MAIN ---
def main():
    robot_enabled = False

    # Init hardware
    print("Initializing hardware...")
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 100
    left_motors  = [Motor(pca, 0, 1), Motor(pca, 6, 7)]
    right_motors = [Motor(pca, 2, 3), Motor(pca, 4, 5)]

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)
        left  = fwd + steer
        right = fwd - steer
        mx = max(abs(left), abs(right))
        if mx > 1.0:
            left /= mx
            right /= mx
        for m in left_motors:  m.set_speed(left)
        for m in right_motors: m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors:
            m.stop()

    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    prev_error     = 0
    last_stop_time = 0

    print("\n--- READY ---")
    print("  SPACE  ->  start / stop robot")
    print("  Q      ->  quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            # --- KEYBOARD (1 ms poll) ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord(' '):
                robot_enabled = not robot_enabled
                print(f"Robot {'STARTED' if robot_enabled else 'STOPPED'}")
                if not robot_enabled:
                    stop_all()

            # --- HOLD if not enabled ---
            if not robot_enabled:
                display = frame.copy()
                cv2.putText(display, "PRESS SPACE TO START",
                            (60, CAMERA_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Robot", display)
                continue

            # --- INFERENCE ---
            results = model.predict(source=frame, conf=0.4, imgsz=640, verbose=False)
            result  = results[0]
            boxes   = result.boxes

            best_y_x   = None
            best_w_x   = None
            max_y_area = 0
            max_w_area = 0
            stop_requested = False
            current_time   = time.time()

            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()

                if cls == 'redline':
                    if y > STOP_THRESHOLD_Y:
                        if (current_time - last_stop_time) > STOP_COOLDOWN:
                            stop_requested = True

                if y < CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF:
                    continue

                area = w * h
                if cls == 'yellowline' and area > max_y_area:
                    max_y_area = area
                    best_y_x = x
                elif cls == 'whiteline' and area > max_w_area:
                    max_w_area = area
                    best_w_x = x

            if stop_requested:
                continue

            # --- TARGET & PD ---
            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2
            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH_PIXELS / 2)
            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH_PIXELS / 2)
            else:
                target_x = CENTER_X

            error      = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error
            steering   = (error * Kp) + (derivative * Kd)

            set_drive(BASE_SPEED, steering)

            # --- DEBUG OVERLAY ---
            annotated = result.plot()
            debug_lines = [
                f"best_w_x: {best_w_x}",
                f"best_y_x: {best_y_x}",
                f"target_x: {target_x:.1f}",
                f"error:    {error:.1f}",
                f"steering: {steering:.4f}",
            ]
            for i, line in enumerate(debug_lines):
                ypos = 25 + i * 22
                cv2.putText(annotated, line, (10, ypos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line, (10, ypos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            debug_y = int(CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF) + 20
            cv2.circle(annotated, (int(target_x), debug_y), 10, (0, 255, 0), -1)
            cv2.line(annotated, (int(CENTER_X), 0), (int(CENTER_X), CAMERA_HEIGHT),
                     (255, 255, 255), 1)

            cv2.imshow("Robot", annotated)

    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping.")
    finally:
        stop_all()
        cap.release()
        cv2.destroyAllWindows()
        print("Motors stopped. Exited cleanly.")


if __name__ == "__main__":
    main()
