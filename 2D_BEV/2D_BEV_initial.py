import depthai as dai
import numpy as np
import cv2
import math

# Constants
image_height = 400
image_width = 640
fx = 400
fy = 400
cx = image_width // 2
cy = image_height // 2

bev_size = 4.0  # meters
bev_resolution = 0.1  # meters per cell
bev_cells = int(bev_size / bev_resolution)
bev_map = np.zeros((bev_cells, bev_cells), dtype=np.uint8)

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.StereoDepth)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
colorCam = pipeline.create(dai.node.ColorCamera)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cam.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
monoLeft.out.link(cam.left)
monoRight.out.link(cam.right)
cam.depth.link(xoutDepth.input)

colorCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
colorCam.setPreviewSize(image_width, image_height)
colorCam.preview.link(xoutRgb.input)

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False)
    rgbQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False)

    while True:
        inDepth = depthQueue.get()
        inRgb = rgbQueue.get()

        depthFrame = inDepth.getFrame()
        depthFrame = cv2.resize(depthFrame, (image_width, image_height))
        rgbFrame = inRgb.getCvFrame()

        # Create point cloud
        points = []
        cloud_vis = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for v in range(0, image_height, 2):
            for u in range(0, image_width, 2):
                d = depthFrame[v, u].astype(np.float32)
                if d < 100 or d > 5000:
                    continue
                Z = d / 1000.0  # convert to meters
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append((X, Y, Z))
                cv2.circle(cloud_vis, (u, v), 1, (255, 255, 255), -1)

        bev_map.fill(0)
        for X, Y, Z in points:
            bev_x = int((X + bev_size / 2) / bev_resolution)
            bev_y = int((Z) / bev_resolution)
            if 0 <= bev_x < bev_cells and 0 <= bev_y < bev_cells:
                bev_map[bev_cells - bev_y - 1, bev_x] = 255

        bev_vis = cv2.resize(bev_map, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        bev_vis = cv2.cvtColor(bev_vis, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Bird's Eye View Occupancy Map", bev_vis)
        cv2.imshow("RGB Camera", rgbFrame)
        cv2.imshow("Point Cloud Preview (2D mask)", cloud_vis)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
