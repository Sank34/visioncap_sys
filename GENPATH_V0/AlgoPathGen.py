import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from collections import deque

# Constants
image_height = 400
image_width = 640
grid_rows = 40
grid_cols = 40
cell_height = image_height // grid_rows
cell_width = image_width // grid_cols
fy = 400
cy = image_height // 2
height_threshold = 350 # mm
N = 20

# Pathfinding (BFS)
def bfs(grid, start, goal):
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True

    while queue:
        node = queue.popleft()
        if node == goal:
            break

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                parent[(nr, nc)] = node
                queue.append((nr, nc))

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent.get(node)
        if node is None: return []
    path.append(start)
    path.reverse()
    return path

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.StereoDepth)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
color = pipeline.create(dai.node.ColorCamera)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cam.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
monoLeft.out.link(cam.left)
monoRight.out.link(cam.right)
cam.depth.link(xoutDepth.input)

color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setInterleaved(False)
color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
color.setPreviewSize(640, 400)
color.preview.link(xoutRgb.input)

imu = pipeline.create(dai.node.IMU)
xoutImu = pipeline.create(dai.node.XLinkOut)
xoutImu.setStreamName("imu")
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xoutImu.input)

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False)
    imuQueue = device.getOutputQueue("imu", maxSize=1, blocking=False)
    rgbQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False)

    while True:
        inDepth = depthQueue.get()
        inImu = imuQueue.get()
        inRgb = rgbQueue.get()

        depthFrame = inDepth.getFrame()
        depthFrame = cv2.resize(depthFrame, (image_width, image_height))
        depthColor = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depthColor = cv2.applyColorMap(depthColor, cv2.COLORMAP_JET)

        rgbFrame = inRgb.getCvFrame()

        imuPacket = inImu.packets[-1]
        acc = imuPacket.acceleroMeter
        ax, ay, az = acc.x, acc.y, acc.z
        pitch_rad = math.atan2(-ax, math.sqrt(ay**2 + az**2))

        bottom_strip = depthFrame[-N:, :]
        valid = bottom_strip[(bottom_strip > 100) & (bottom_strip < 5000)]
        if len(valid) == 0:
            continue
        ground_depth = np.median(valid)
        camera_height_mm = ground_depth * abs(math.sin(pitch_rad))

        occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
        height_map = np.zeros((grid_rows, grid_cols))

        for i in range(grid_rows):
            for j in range(grid_cols):
                cell = depthFrame[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                valid_depths = cell[(cell > 100) & (cell < 5000)]
                if len(valid_depths) == 0:
                    continue
                cell_depth = np.percentile(valid_depths, 10)
                v = i * cell_height + cell_height // 2
                theta_p = math.atan2((v - cy), fy)
                object_height = camera_height_mm - cell_depth * math.sin(pitch_rad + theta_p)
                object_height = max(0, object_height)
                height_map[i, j] = object_height
                occupancy_grid[i, j] = 1 if object_height > height_threshold else 0

        # --- Pathfinding ---
        start = (grid_rows - 1, grid_cols // 2)
        goal = (0, grid_cols // 2)
        path = bfs(occupancy_grid, start, goal)
        for r, c in path:
            occupancy_grid[r, c] = 2

        occ_vis = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
        occ_vis[occupancy_grid == 0] = [0, 0, 0]
        occ_vis[occupancy_grid == 1] = [255, 255, 255]
        occ_vis[occupancy_grid == 2] = [0, 255, 0]
        occ_vis = cv2.resize(occ_vis, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack((depthColor, occ_vis))
        cv2.imshow("Depth | Occupancy Grid + Path", combined)
        cv2.imshow("RGB Camera", rgbFrame)

        vis_height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_height_map = cv2.resize(vis_height_map, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        vis_height_map = cv2.applyColorMap(vis_height_map, cv2.COLORMAP_JET)
        cv2.imshow("Object Height (mm)", vis_height_map)

        strip_vis = cv2.normalize(bottom_strip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        strip_vis = cv2.applyColorMap(strip_vis, cv2.COLORMAP_JET)
        strip_vis = cv2.resize(strip_vis, (image_width, 50))
        cv2.imshow("Ground Estimation Strip", strip_vis)

        print(f"Pitch: {math.degrees(pitch_rad):.2f}°, Camera Height: {camera_height_mm:.0f} mm")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
