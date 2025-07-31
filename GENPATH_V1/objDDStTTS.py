import depthai as dai
import blobconverter
import numpy as np
import cv2
import time
import pyttsx3

# Config
GRID_WIDTH = 5
GRID_HEIGHT = 5
CELL_SIZE = 100
X_RANGE = (-1.5, 1.5)  # meters
Z_RANGE = (0.3, 3.0)   # meters
DIST_THRESHOLD = 5.0
MEMORY_DURATION = 2.0  # seconds

label_map = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

low_risk_labels = {"bottle", "cat", "pottedplant", "bird"}

def real_world_to_grid(x, z):
    if not (X_RANGE[0] <= x <= X_RANGE[1] and Z_RANGE[0] <= z <= Z_RANGE[1]):
        return None
    col = int((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * GRID_WIDTH)
    row = int((z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]) * GRID_HEIGHT)
    col = max(0, min(GRID_WIDTH - 1, col)) # col act
    row = max(0, min(GRID_HEIGHT - 1, row)) # row act
    return row, col

# Init TTS engine
engine = pyttsx3.init()
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')  # macOS-specific voice
engine.setProperty('rate', 180)  # voice speed rate

# Pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

conf1 = 200
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(conf1)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

conf2 = 0.5
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
nn.setConfidenceThreshold(conf2)
cam_rgb.preview.link(nn.input)

xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.depth.link(xout_depth.input)

# Connect to camera
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
    fx = intrinsics[0][0]
    cx_cam = intrinsics[0][2]
    cy_cam = intrinsics[1][2]

    detection_memory = [[0.0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    last_spoken_direction = ""  # For TTS direction change tracking

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        now = time.time()

        in_video = q_video.get()
        in_nn = q_nn.get()
        in_depth = q_depth.get()

        frame = in_video.getCvFrame()
        depth_map = in_depth.getFrame()
        detections = in_nn.detections

        for det in detections:
            cx_norm = (det.xmin + det.xmax) / 2
            cy_norm = det.ymax - 0.05
            cx = int(cx_norm * depth_map.shape[1])
            cy = int(cy_norm * depth_map.shape[0])
            cx = np.clip(cx, 0, depth_map.shape[1] - 1)
            cy = np.clip(cy, 0, depth_map.shape[0] - 1)

            depth_mm = depth_map[cy, cx]
            if depth_mm == 0 or depth_mm > 10000:
                continue

            depth_m = depth_mm / 1000.0
            label = label_map[det.label] if det.label < len(label_map) else str(det.label)
            conf = int(det.confidence * 100)

            is_low_risk = (
                label in low_risk_labels or
                depth_m > DIST_THRESHOLD
            )

            color = (0, 255, 0) if is_low_risk else (0, 0, 255)

            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf}%) | {depth_m:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if not is_low_risk:
                real_x = (cx - cx_cam) * depth_m / fx
                real_z = depth_m
                coords = real_world_to_grid(real_x, real_z)
                if coords:
                    r, c = coords
                    detection_memory[r][c] = now

        occupancy_matrix = [
            [1 if now - detection_memory[r][c] < MEMORY_DURATION else 0 for c in range(GRID_WIDTH)]
            for r in range(GRID_HEIGHT)
        ]

        column_scores = []
        for col in range(GRID_WIDTH):
            score = 0
            for row in range(GRID_HEIGHT - 1, -1, -1):
                if occupancy_matrix[row][col] == 0:
                    score += 1
                else:
                    break
            column_scores.append(score)

        max_score = max(column_scores)
        best_columns = [i for i, score in enumerate(column_scores) if score == max_score]
        center_col = GRID_WIDTH // 2
        best_col = min(best_columns, key=lambda c: abs(c - center_col))

        if best_col < center_col:
            direction = "LEFT"
        elif best_col > center_col:
            direction = "RIGHT"
        else:
            direction = "FORWARD"

        if direction != last_spoken_direction:
            engine.say(f"Go {direction.lower()}")
            engine.runAndWait()
            last_spoken_direction = direction

        grid_img = np.ones((GRID_HEIGHT * CELL_SIZE, GRID_WIDTH * CELL_SIZE, 3), dtype=np.uint8) * 255
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                time_diff = now - detection_memory[row][col]
                is_occupied = time_diff < MEMORY_DURATION
                cell_color = (0, 0, 255) if is_occupied else (0, 255, 0)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), cell_color, -1)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

                if is_occupied:
                    seconds_left = MEMORY_DURATION - time_diff
                    countdown_text = f"{seconds_left:.1f}s"
                    text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = x1 + (CELL_SIZE - text_size[0]) // 2
                    text_y = y1 + (CELL_SIZE + text_size[1]) // 2
                    cv2.putText(grid_img, countdown_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for row in range(GRID_HEIGHT - 1, -1, -1):
            if occupancy_matrix[row][best_col] == 0:
                x1 = best_col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (180, 180, 180), -1)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            else:
                break

        cv2.putText(grid_img, f"Go: {direction}", (10, GRID_HEIGHT * CELL_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

        resized_video = cv2.resize(frame, (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
        stacked = np.vstack((resized_video, grid_img))
        cv2.imshow("VisionCap: Pathfinding", stacked)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

# matrix stable -> if user moves(IMU) -> refresh the whole matrix