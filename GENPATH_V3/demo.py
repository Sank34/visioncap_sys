import depthai as dai
import numpy as np
import cv2

# Config
ROI_NAMES = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_ROIS = len(ROI_NAMES)
LOW_RISK_LABELS = {'bottle', 'cat', 'pottedplant', 'bird'}
DISTANCE_THRESHOLD = 3.0
DECISION_PRIORITY = ['center', 'left', 'right', 'far_left', 'far_right']
ALLOWED_LABELS = {0, 39}  # person and bottle

label_map = [
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


DEPTH_DIFF_THRESHOLD = 5  # threshold for detecting movement
FREEZE_DURATION = 1000  # number of static frames before freezing

# Arrow display
def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    arrow_color = (0, 255, 255)
    thickness = 4
    length = 100
    center_x = w // 2
    y_start = h - 30
    tip_offset = {
        'LEFT': (-length, 0),
        'RIGHT': (length, 0),
        'CENTER': (0, -length),
        'FAR_LEFT': (-length - 30, 0),
        'FAR_RIGHT': (length + 30, 0),
        'STOP': (0, 0)
    }
    if direction != 'STOP':
        dx, dy = tip_offset.get(direction, (0, -length))
        end_point = (center_x + dx, y_start + dy)
        cv2.arrowedLine(frame, (center_x, y_start), end_point, arrow_color, thickness, tipLength=0.4)
    else:
        cv2.line(frame, (center_x - 20, y_start - 20), (center_x + 20, y_start + 20), (0, 0, 255), thickness)
        cv2.line(frame, (center_x - 20, y_start + 20), (center_x + 20, y_start - 20), (0, 0, 255), thickness)

# Pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(640, 352)
cam_rgb.setInterleaved(False)

mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

depth = pipeline.createStereoDepth()
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

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.depth.link(xout_depth.input)

# Run
with dai.Device(pipeline) as device:
    q_nn = device.getOutputQueue("nn")
    q_video = device.getOutputQueue("video")
    q_depth = device.getOutputQueue("depth")

    prev_depth = None
    still_frames = 0
    mode = "SCANNING"

    while True:
        frame = q_video.get().getCvFrame()
        detections = q_nn.get().detections
        depth_frame = q_depth.get().getFrame()

        # --- Motion detection logic ---
        if prev_depth is not None:
            diff = cv2.absdiff(depth_frame, prev_depth)
            mean_diff = np.mean(diff)
            if mean_diff < DEPTH_DIFF_THRESHOLD:
                still_frames += 1
            else:
                still_frames = 0
            mode = "FREEZE" if still_frames >= FREEZE_DURATION else "SCANNING"
        prev_depth = depth_frame.copy()

        roi_stats = {name: [] for name in ROI_NAMES}
        roi_width = frame.shape[1] // NUM_ROIS

        for det in detections:
            label_id = det.label
            if label_id not in ALLOWED_LABELS:
                continue
            label = label_map[label_id]
            x1_norm, x2_norm = det.xmin, det.xmax
            cx = int(((x1_norm + x2_norm) / 2) * frame.shape[1])
            cy = int((det.ymax - 0.05) * frame.shape[0])
            depth_mm = depth_frame[min(cy, depth_frame.shape[0]-1), min(cx, depth_frame.shape[1]-1)]
            if depth_mm == 0 or depth_mm > 10000:
                continue
            depth_m = depth_mm / 1000.0
            risk = "low" if label in LOW_RISK_LABELS else "high"
            pixel_x1 = int(x1_norm * frame.shape[1])
            pixel_x2 = int(x2_norm * frame.shape[1])
            start_roi = min(NUM_ROIS - 1, pixel_x1 // roi_width)
            end_roi = min(NUM_ROIS - 1, pixel_x2 // roi_width)
            for roi_index in range(start_roi, end_roi + 1):
                roi_stats[ROI_NAMES[roi_index]].append((label, risk, depth_m))

            draw_x1 = pixel_x1
            draw_y1 = int(det.ymin * frame.shape[0])
            draw_x2 = pixel_x2
            draw_y2 = int(det.ymax * frame.shape[0])
            color = (0, 255, 0) if risk == "low" else (0, 0, 255)
            cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
            cv2.putText(frame, f"{label} {depth_m:.1f}m", (draw_x1, draw_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        for i in range(1, NUM_ROIS):
            x = i * roi_width
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)

        scores = {}
        for name, objects in roi_stats.items():
            high_risk_count = sum(1 for obj in objects if obj[1] == 'high' and obj[2] < DISTANCE_THRESHOLD)
            scores[name] = high_risk_count

        print(f"\n[DECISION SCORES] - MODE: {mode}")
        for name in ROI_NAMES:
            print(f"{name.upper()}: {scores[name]} high-risk near")

        direction = "STOP"
        for candidate in DECISION_PRIORITY:
            if scores[candidate] == 0:
                direction = candidate.upper()
                break

        cv2.putText(frame, f"Decision: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        draw_arrow(frame, direction)

        # Mode display (SCANNING/FREEZE)
        color = (0, 255, 0) if mode == "SCANNING" else (255, 255, 0)
        cv2.putText(frame, f"Mode: {mode}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show windows
        cv2.imshow("ROIs with Decision", frame)
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Depth Heatmap", depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()