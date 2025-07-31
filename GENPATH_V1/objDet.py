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
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # High-res output
cam_rgb.setPreviewSize(300, 300)  # For NN input
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Detection Network
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
detection_nn.setConfidenceThreshold(0.9)

# Link preview to NN
cam_rgb.preview.link(detection_nn.input)

# XLink outputs
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# ------------------------
# Run pipeline
# ------------------------
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue("video", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        in_video = q_video.get()
        in_nn = q_nn.get()

        frame = in_video.getCvFrame()
        detections = in_nn.detections

        for detection in detections:
            # Normalize and draw bounding box
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # Get class label
            label = label_map[detection.label] if detection.label < len(label_map) else str(detection.label)
            conf = int(detection.confidence * 100)

            # Draw box + label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf}%)", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("High-Res Object Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()