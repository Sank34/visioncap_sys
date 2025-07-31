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
X_RANGE = (-1.5, 1.5)
Z_RANGE = (0.3, 3.0)
DIST_THRESHOLD = 5.0
MEMORY_DURATION = 2.0  # seconds

label_map = [  # COCO
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
low_risk_labels = {"bottle", "cat", "pottedplant", "bird"}

def real_world_to_grid(x, z):
    if not (X_RANGE[0] <= x <= X_RANGE[1] and Z_RANGE[0] <= z <= Z_RANGE[1]):
        return None
    col = int((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * GRID_WIDTH)
    row = int((z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]) * GRID_HEIGHT)
    return max(0, min(GRID_HEIGHT - 1, row)), max(0, min(GRID_WIDTH - 1, col))

# TTS
engine = pyttsx3.init()
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
engine.setProperty('rate', 180)

# Pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(640, 352)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

nn = pipeline.createYoloDetectionNetwork()
nn.setBlobPath("/Users/sanki/PycharmProjects/TestOccupancyGrid/GENPATH_V1/YOLOv8n_coco_640x352.blob")
nn.setConfidenceThreshold(0.5)
nn.setNumClasses(80)
nn.setCoordinateSize(4)
nn.setAnchors([
    10, 13, 16, 30, 33, 23,
    30, 61, 62, 45, 59, 119,
    116, 90, 156, 198, 373, 326
])
nn.setAnchorMasks({
    "side80": [0, 1, 2],
    "side40": [3, 4, 5],
    "side20": [6, 7, 8]
})
nn.setIouThreshold(0.5)
cam_rgb.preview.link(nn.input)

# XLinkOuts
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.depth.link(xout_depth.input)

# Run pipeline
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video")
    q_nn = device.getOutputQueue("nn")
    q_depth = device.getOutputQueue("depth")

    calib = device.readCalibration()
    intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 352)
    fx, cx_cam, cy_cam = intr[0][0], intr[0][2], intr[1][2]

    memory = [[0.0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    last_dir = ""

    while True:
        now = time.time()
        frame = q_video.get().getCvFrame()
        detections = q_nn.get().detections
        depth_map = q_depth.get().getFrame()

        for det in detections:
            cx = int(((det.xmin + det.xmax) / 2) * depth_map.shape[1])
            cy = int((det.ymax - 0.05) * depth_map.shape[0])
            cx = np.clip(cx, 0, depth_map.shape[1] - 1)
            cy = np.clip(cy, 0, depth_map.shape[0] - 1)

            depth_mm = depth_map[cy, cx]
            label = label_map[det.label] if det.label < len(label_map) else str(det.label)

            if depth_mm == 0 or depth_mm > 10000:
                continue

            depth_m = depth_mm / 1000.0
            conf = int(det.confidence * 100)
            is_safe = (label in low_risk_labels or depth_m > DIST_THRESHOLD)
            color = (0, 255, 0) if is_safe else (0, 0, 255)

            # Draw box
            x1, y1, x2, y2 = [
                int(val * frame.shape[1] if i % 2 == 0 else val * frame.shape[0])
                for i, val in enumerate([det.xmin, det.ymin, det.xmax, det.ymax])
            ]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf}%) | {depth_m:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if not is_safe:
                real_x = (cx - cx_cam) * depth_m / fx
                real_z = depth_m
                coords = real_world_to_grid(real_x, real_z)
                if coords:
                    r, c = coords
                    memory[r][c] = now

        # Occupancy matrix
        occ = [
            [1 if now - memory[r][c] < MEMORY_DURATION else 0 for c in range(GRID_WIDTH)]
            for r in range(GRID_HEIGHT)
        ]

        # Column scores
        col_scores = []
        for c in range(GRID_WIDTH):
            score = 0
            for r in range(GRID_HEIGHT - 1, -1, -1):
                if occ[r][c] == 0:
                    score += 1
                else:
                    break
            col_scores.append(score)

        max_score = max(col_scores)
        center = GRID_WIDTH // 2
        best_cols = [c for c, s in enumerate(col_scores) if s == max_score]
        best_col = min(best_cols, key=lambda c: abs(c - center))

        # Corrected direction logic
        if best_col == center and col_scores[center] == max_score and col_scores[center] > 0:
            direction = "FORWARD"
        elif best_col < center:
            direction = "LEFT"
        elif best_col > center:
            direction = "RIGHT"
        else:
            direction = "FORWARD"

        # TTS
        if direction != last_dir:
            engine.say(f"Go {direction.lower()}")
            engine.runAndWait()
            last_dir = direction

        # Grid image
        grid_img = np.ones((GRID_HEIGHT * CELL_SIZE, GRID_WIDTH * CELL_SIZE, 3), np.uint8) * 255
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                diff = now - memory[r][c]
                filled = diff < MEMORY_DURATION
                color = (0, 0, 255) if filled else (0, 255, 0)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                if filled:
                    txt = f"{MEMORY_DURATION - diff:.1f}s"
                    sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.putText(grid_img, txt, (x1 + (CELL_SIZE - sz[0]) // 2, y1 + (CELL_SIZE + sz[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Gray path
        for r in range(GRID_HEIGHT - 1, -1, -1):
            if occ[r][best_col] == 0:
                x1 = best_col * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (180, 180, 180), -1)
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            else:
                break

        cv2.putText(grid_img, f"Go: {direction}", (10, GRID_HEIGHT * CELL_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

        # Dual view
        small = cv2.resize(frame, (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
        preview = cv2.resize(frame, (640, 704))
        stacked = np.vstack((small, cv2.resize(grid_img, (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))))
        padded_left = cv2.copyMakeBorder(stacked, 0, 0, 0, 640 - stacked.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        preview_resized = cv2.resize(preview, (640, padded_left.shape[0]))
        full_display = np.hstack((padded_left, preview_resized))

        cv2.imshow("VisionCap: YOLOv8 Navigation - Dual Preview", full_display)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()