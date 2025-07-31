import depthai as dai
import numpy as np
import cv2

# Config
ROI_NAMES = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_ROIS = len(ROI_NAMES)
LOW_RISK_LABELS = {'bottle', 'cat', 'pottedplant', 'bird'}
DISTANCE_THRESHOLD = 5.0  # max effective range (in meters)
DECISION_PRIORITY = ['center', 'left', 'right', 'far_left', 'far_right']
EPSILON = 1e-2  # To avoid division by 0 in weights

label_map = [...]  # same label list as before

# Pipeline setup
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

xout_nn = pipeline.createXLinkOut(); xout_nn.setStreamName("nn"); nn.out.link(xout_nn.input)
xout_video = pipeline.createXLinkOut(); xout_video.setStreamName("video"); cam_rgb.video.link(xout_video.input)
xout_depth = pipeline.createXLinkOut(); xout_depth.setStreamName("depth"); depth.depth.link(xout_depth.input)

# Run
with dai.Device(pipeline) as device:
    q_nn = device.getOutputQueue("nn")
    q_video = device.getOutputQueue("video")
    q_depth = device.getOutputQueue("depth")

    while True:
        frame = q_video.get().getCvFrame()
        detections = q_nn.get().detections
        depth_map = q_depth.get().getFrame()

        roi_stats = {name: [] for name in ROI_NAMES}
        roi_scores = {name: 0.0 for name in ROI_NAMES}
        roi_width = frame.shape[1] // NUM_ROIS

        for det in detections:
            label_id = det.label
            label = label_map[label_id] if label_id < len(label_map) else str(label_id)

            cx = int(((det.xmin + det.xmax) / 2) * frame.shape[1])
            cy = int((det.ymax - 0.05) * frame.shape[0])
            depth_mm = depth_map[min(cy, depth_map.shape[0]-1), min(cx, depth_map.shape[1]-1)]
            if depth_mm == 0 or depth_mm > 10000:
                continue

            depth_m = depth_mm / 1000.0
            risk = "low" if label in LOW_RISK_LABELS else "high"
            roi_index = min(NUM_ROIS - 1, cx // roi_width)
            roi_name = ROI_NAMES[roi_index]
            roi_stats[roi_name].append((label, risk, depth_m))

            if risk == "high" and depth_m <= DISTANCE_THRESHOLD:
                roi_scores[roi_name] += 1.0 / (depth_m**2 + EPSILON)

            x1, y1 = int(det.xmin * frame.shape[1]), int(det.ymin * frame.shape[0])
            x2, y2 = int(det.xmax * frame.shape[1]), int(det.ymax * frame.shape[0])
            color = (0, 255, 0) if risk == "low" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {depth_m:.1f}m", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw ROI grid + score
        for i in range(1, NUM_ROIS):
            x = i * roi_width
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)

        for i, name in enumerate(ROI_NAMES):
            score = roi_scores[name]
            text = f"{score:.2f}"
            text_x = i * roi_width + roi_width // 2 - 30
            text_y = frame.shape[0] - 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        # Decide direction: choose ROI with minimum score
        best_roi = min(ROI_NAMES, key=lambda name: roi_scores[name])
        if roi_scores[best_roi] > 0.5:
            decision = "STOP"
        else:
            decision = best_roi.upper()

        cv2.putText(frame, f"Decision: {decision}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ROIs with Weighted Decision", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()