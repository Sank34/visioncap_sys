import depthai as dai
import blobconverter
import cv2
import numpy as np

# ------------------------
# Label map for MobileNet-SSD
# ------------------------
label_map = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Define low-risk object labels (everything else is high-risk)
low_risk_labels = {"bottle", "cat", "pottedplant", "tvmonitor", "bird"}

# ------------------------
# Pipeline setup
# ------------------------
pipeline = dai.Pipeline()

# Color Camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Mono cameras for stereo depth
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Depth
conf1 = 200 # depth conf thresh
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(conf1)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

# Detection
conf2 = 0.9 # obj conf thresh
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
detection_nn.setConfidenceThreshold(conf2)
cam_rgb.preview.link(detection_nn.input)

# XLink Outputs
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.depth.link(xout_depth.input)

# ------------------------
# Run pipeline
# ------------------------
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        in_video = q_video.get()
        in_nn = q_nn.get()
        in_depth = q_depth.get()

        frame = in_video.getCvFrame()
        depth_map = in_depth.getFrame()
        detections = in_nn.detections

        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            x1, y1, x2, y2 = bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Depth in meters
            depth_mm = depth_map[cy, cx].astype(np.uint16)
            depth_m = depth_mm / 1000.0 if depth_mm > 0 else 0

            # Class label and confidence
            label = label_map[detection.label] if detection.label < len(label_map) else str(detection.label)
            conf = int(detection.confidence * 100)

            # New risk classification: based on label + depth
            if label in low_risk_labels:
                is_low_risk = True
            elif depth_m > 2.0:
                is_low_risk = True
            else:
                is_low_risk = False

            color = (0, 255, 0) if is_low_risk else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf}%) | {depth_m:.2f}m"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Object Detection + Depth + Risk", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()