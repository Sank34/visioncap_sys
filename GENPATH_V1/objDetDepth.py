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

# Mono cameras (required for depth)
mono_left = pipeline.createMonoCamera()
mono_right = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Depth node
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB camera

mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

# Detection network
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
detection_nn.setConfidenceThreshold(0.5)
cam_rgb.preview.link(detection_nn.input)

# XLink outputs
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
        depth_frame = in_depth.getFrame()
        depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_frame = cv2.convertScaleAbs(depth_frame)

        detections = in_nn.detections

        for detection in detections:
            # Get bounding box in pixel coords
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            x1, y1, x2, y2 = bbox
            label = label_map[detection.label] if detection.label < len(label_map) else str(detection.label)
            conf = int(detection.confidence * 100)

            # Get center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Get depth in millimeters
            depth_raw = in_depth.getFrame()[cy, cx].astype(np.uint16)
            depth_meters = depth_raw / 1000.0 if depth_raw > 0 else 0  # Avoid 0 division

            # Draw everything
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf}%) | {depth_meters:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Object Detection + Depth", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()