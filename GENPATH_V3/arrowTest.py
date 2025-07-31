import depthai as dai
import numpy as np
import cv2
#GOOD

# Config
ROI_NAMES = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_ROIS = len(ROI_NAMES)
LOW_RISK_LABELS = {'bottle', 'cat', 'pottedplant', 'bird'}  # ONLY LOW-RISK; others are high
DISTANCE_THRESHOLD = 3.0  # meters
DECISION_PRIORITY = ['center', 'left', 'right', 'far_left', 'far_right']

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

# add depth preview heatmap
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
        # Draw red stop sign (X)
        cv2.line(frame, (center_x - 20, y_start - 20), (center_x + 20, y_start + 20), (0, 0, 255), thickness)
        cv2.line(frame, (center_x - 20, y_start + 20), (center_x + 20, y_start - 20), (0, 0, 255), thickness)

# Run
with dai.Device(pipeline) as device:
    q_nn = device.getOutputQueue("nn")
    q_video = device.getOutputQueue("video")
    q_depth = device.getOutputQueue("depth")

    intrinsics = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 352)
    fx, cx = intrinsics[0][0], intrinsics[0][2]

    while True:
        frame = q_video.get().getCvFrame()
        detections = q_nn.get().detections
        depth_frame = q_depth.get().getFrame()

        roi_stats = {name: [] for name in ROI_NAMES}
        roi_width = frame.shape[1] // NUM_ROIS

        ALLOWED_LABELS = {0,39}
        for det in detections:
            label_id = det.label
            if label_id not in ALLOWED_LABELS:
                continue
            label = label_map[label_id] if label_id < len(label_map) else str(label_id)

            x1_norm = det.xmin
            x2_norm = det.xmax
            cx = int(((x1_norm + x2_norm) / 2) * frame.shape[1])
            cy = int((det.ymax - 0.05) * frame.shape[0])
            depth_mm = depth_frame[min(cy, depth_frame.shape[0] - 1), min(cx, depth_frame.shape[1] - 1)]

            if depth_mm == 0 or depth_mm > 10000:
                continue

            depth_m = depth_mm / 1000.0
            risk = "low" if label in LOW_RISK_LABELS else "high"
            pixel_x1 = int(x1_norm * frame.shape[1])
            pixel_x2 = int(x2_norm * frame.shape[1])

            # Assign object to all overlapping ROIs
            start_roi = min(NUM_ROIS - 1, pixel_x1 // roi_width)
            end_roi = min(NUM_ROIS - 1, pixel_x2 // roi_width)

            for roi_index in range(start_roi, end_roi + 1):
                roi_name = ROI_NAMES[roi_index]
                roi_stats[roi_name].append((label, risk, depth_m))

            # Draw bounding box
            draw_x1 = pixel_x1
            draw_y1 = int(det.ymin * frame.shape[0])
            draw_x2 = pixel_x2
            draw_y2 = int(det.ymax * frame.shape[0])
            color = (0, 255, 0) if risk == "low" else (0, 0, 255)
            cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
            cv2.putText(frame, f"{label} {depth_m:.1f}m", (draw_x1, draw_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Draw ROI grid lines
        for i in range(1, NUM_ROIS):
            x = i * roi_width
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)

        # Decision logic
        scores = {}
        for name, objects in roi_stats.items():
            high_risk_count = sum(1 for obj in objects if obj[1] == 'high' and obj[2] < DISTANCE_THRESHOLD)
            scores[name] = high_risk_count

        print("\n[DECISION SCORES]")
        for name in ROI_NAMES:
            print(f"{name.upper()}: {scores[name]} high-risk near")

        direction = "STOP"
        for candidate in DECISION_PRIORITY:
            if scores[candidate] == 0:
                direction = candidate.upper()
                break

        cv2.putText(frame, f"Decision: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        draw_arrow(frame, direction)
        cv2.imshow("ROIs with Decision", frame)
        depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Depth Heatmap", depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()