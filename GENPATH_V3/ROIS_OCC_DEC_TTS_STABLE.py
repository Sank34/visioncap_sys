import depthai as dai
import numpy as np
import cv2
import pyttsx3
import time

# === CONFIG ===
ROI_NAMES = ['far_left', 'left', 'center', 'right', 'far_right']
NUM_ROIS = len(ROI_NAMES)
LOW_RISK_LABELS = {'bottle', 'cat', 'pottedplant', 'bird'}
DISTANCE_THRESHOLD = 3.0
DECISION_PRIORITY = ['center', 'left', 'right', 'far_left', 'far_right']
SPEAK_COOLDOWN = 3.0  # seconds

# === TTS SETUP ===
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')

last_spoken_time = 0
last_direction = ""

# === LABEL MAP ===
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

# === PIPELINE SETUP ===
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

# === RUN LOOP ===
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

        for det in detections:
            label_id = det.label
            label = label_map[label_id] if label_id < len(label_map) else str(label_id)
            x1 = int(det.xmin * frame.shape[1])
            x2 = int(det.xmax * frame.shape[1])
            cx = (x1 + x2) // 2
            cy = int((det.ymax - 0.05) * frame.shape[0])
            depth_mm = depth_frame[min(cy, depth_frame.shape[0]-1), min(cx, depth_frame.shape[1]-1)]

            if depth_mm == 0 or depth_mm > 10000:
                continue

            depth_m = depth_mm / 1000.0
            risk = "low" if label in LOW_RISK_LABELS else "high"
            start_roi = min(NUM_ROIS - 1, x1 // roi_width)
            end_roi = min(NUM_ROIS - 1, x2 // roi_width)

            for i in range(start_roi, end_roi + 1):
                roi_stats[ROI_NAMES[i]].append((label, risk, depth_m))

            color = (0, 255, 0) if risk == "low" else (0, 0, 255)
            cv2.rectangle(frame, (x1, int(det.ymin * frame.shape[0])), (x2, int(det.ymax * frame.shape[0])), color, 2)
            cv2.putText(frame, f"{label} {depth_m:.1f}m", (x1, int(det.ymin * frame.shape[0]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for i in range(1, NUM_ROIS):
            x = i * roi_width
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)

        scores = {name: sum(1 for obj in objs if obj[1] == 'high' and obj[2] < DISTANCE_THRESHOLD)
                  for name, objs in roi_stats.items()}

        direction = "STOP"
        for name in DECISION_PRIORITY:
            if scores[name] == 0:
                direction = name.upper()
                break

        now = time.time()
        if direction != last_direction and now - last_spoken_time > SPEAK_COOLDOWN:
            engine.say(f"Go {direction.lower()}")
            engine.runAndWait()
            last_spoken_time = now
            last_direction = direction

        cv2.putText(frame, f"Decision: {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("ROIs with Decision", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()