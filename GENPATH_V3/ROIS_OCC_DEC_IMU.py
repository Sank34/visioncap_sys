import depthai as dai
import numpy as np
import cv2
import time
from enum import Enum

# Config
ROI_NAMES = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_ROIS = len(ROI_NAMES)
LOW_RISK_LABELS = {'bottle', 'cat', 'pottedplant', 'bird'}
DISTANCE_THRESHOLD = 3.0
DECISION_PRIORITY = ['center', 'left', 'right', 'far_left', 'far_right']
MOVEMENT_THRESHOLD = 0.5
FREEZE_TIMEOUT = 5.0

class NavState(Enum):
    SCANNING = 1
    FROZEN = 2

label_map = [...]  # keep your label_map list here

pipeline = dai.Pipeline()

# Cameras
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

# Depth
depth = pipeline.createStereoDepth()
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

# NN
nn = pipeline.createYoloDetectionNetwork()
nn.setBlobPath("/Users/sanki/PycharmProjects/TestOccupancyGrid/GENPATH_V1/YOLOv8n_coco_640x352.blob")
nn.setConfidenceThreshold(0.5)
nn.setNumClasses(80)
nn.setCoordinateSize(4)
nn.setAnchors([
    10,13,16,30,33,23, 30,61,62,45,59,119, 116,90,156,198,373,326
])
nn.setAnchorMasks({"side80": [0,1,2], "side40": [3,4,5], "side20": [6,7,8]})
nn.setIouThreshold(0.5)
cam_rgb.preview.link(nn.input)

# IMU
imu = pipeline.createIMU()
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

# Outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.depth.link(xout_depth.input)

xout_imu = pipeline.createXLinkOut()
xout_imu.setStreamName("imu")
imu.out.link(xout_imu.input)

# Run
with dai.Device(pipeline) as device:
    q_nn = device.getOutputQueue("nn")
    q_video = device.getOutputQueue("video")
    q_depth = device.getOutputQueue("depth")
    q_imu = device.getOutputQueue("imu")

    intrinsics = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 352)
    fx, cx = intrinsics[0][0], intrinsics[0][2]

    state = NavState.SCANNING
    last_movement_time = time.time()

    while True:
        now = time.time()

        # IMU check
        while q_imu.has():
            accel = q_imu.get().packets[0].acceleroMeter
            mag = np.linalg.norm([accel.x, accel.y, accel.z] - np.array([0, 0, 9.81]))
            if mag > MOVEMENT_THRESHOLD:
                last_movement_time = now
                state = NavState.SCANNING

        if state == NavState.SCANNING and (now - last_movement_time > FREEZE_TIMEOUT):
            state = NavState.FROZEN

        frame = q_video.get().getCvFrame()
        detections = q_nn.get().detections
        depth_frame = q_depth.get().getFrame()

        roi_stats = {name: [] for name in ROI_NAMES}
        roi_width = frame.shape[1] // NUM_ROIS

        if state == NavState.SCANNING:
            for det in detections:
                label_id = det.label
                label = label_map[label_id] if label_id < len(label_map) else str(label_id)
                cx = int(((det.xmin + det.xmax) / 2) * frame.shape[1])
                cy = int((det.ymax - 0.05) * frame.shape[0])
                depth_mm = depth_frame[min(cy, depth_frame.shape[0] - 1), min(cx, depth_frame.shape[1] - 1)]

                if depth_mm == 0 or depth_mm > 10000:
                    continue

                depth_m = depth_mm / 1000.0
                risk = "low" if label in LOW_RISK_LABELS else "high"

                x1 = int(det.xmin * frame.shape[1])
                x2 = int(det.xmax * frame.shape[1])
                roi_start = min(NUM_ROIS - 1, x1 // roi_width)
                roi_end = min(NUM_ROIS - 1, x2 // roi_width)

                for i in range(roi_start, roi_end + 1):
                    roi_stats[ROI_NAMES[i]].append((label, risk, depth_m))

                color = (0, 255, 0) if risk == "low" else (0, 0, 255)
                y1 = int(det.ymin * frame.shape[0])
                y2 = int(det.ymax * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {depth_m:.1f}m", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for i in range(1, NUM_ROIS):
            x = i * roi_width
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)

        scores = {}
        for name, objs in roi_stats.items():
            scores[name] = sum(1 for obj in objs if obj[1] == 'high' and obj[2] < DISTANCE_THRESHOLD)

        direction = "STOP"
        for candidate in DECISION_PRIORITY:
            if scores.get(candidate, 0) == 0:
                direction = candidate.upper()
                break

        cv2.putText(frame, f"Mode: {state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, f"Decision: {direction}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("VisionCap - IMU Based Nav", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()