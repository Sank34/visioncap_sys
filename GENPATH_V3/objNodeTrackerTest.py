import depthai as dai
import blobconverter
import numpy as np
import cv2

# COCO Label map
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

# DepthAI pipeline
pipeline = dai.Pipeline()

# RGB Camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 352)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# YOLOv8 Network
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
cam.preview.link(nn.input)

# Object Tracker Node
tracker = pipeline.createObjectTracker()
tracker.setDetectionLabelsToTrack([0, 1, 2, 5, 7, 15, 16])  # example: person, bicycle, car, bus, truck, cat, dog
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
nn.out.link(tracker.inputDetections)
cam.preview.link(tracker.inputTrackerFrame)

# Output streams
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("video")
cam.preview.link(xout_rgb.input)

xout_track = pipeline.createXLinkOut()
xout_track.setStreamName("tracker")
tracker.out.link(xout_track.input)

# Device runtime
with dai.Device(pipeline) as device:
    video_q = device.getOutputQueue("video")
    track_q = device.getOutputQueue("tracker")

    while True:
        frame = video_q.get().getCvFrame()
        tracked = track_q.get().tracklets

        for track in tracked:
            roi = track.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            label = track.label
            label_text = label_map[label] if label < len(label_map) else str(label)
            tracker_id = track.id
            status = track.status.name

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label_text} ID:{tracker_id} {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            print(f"[TRACKED] ID: {tracker_id} | Label: {label_text} | Status: {status}")

        cv2.imshow("YOLOv8 with Object Tracker", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()