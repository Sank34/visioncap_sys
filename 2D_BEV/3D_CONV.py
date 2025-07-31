import depthai as dai
import numpy as np
import cv2
import math
import open3d as o3d
from collections import deque

image_height = 400
image_width = 640
fx = 400
fy = 400
cx = image_width // 2
cy = image_height // 2

grid_resolution = 0.1
grid_width = 4.0
grid_height = 4.0
cols = int(grid_width / grid_resolution)
rows = int(grid_height / grid_resolution)
occu_grid = np.zeros((rows, cols), dtype=np.uint8)

start_pos = (int(rows * 0.9), cols // 2)
goal_pos = (int(rows * 0.1), cols // 2)

def world_to_grid(x, z):
    col = int((x + grid_width / 2) / grid_resolution)
    row = int((z) / grid_resolution)
    return row, col

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
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = node[0]+dr, node[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                parent[(nr, nc)] = node
                queue.append((nr, nc))

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent.get(node)
        if node is None:
            return []
    path.append(start)
    path.reverse()
    return path

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

        hit_count = np.zeros((rows, cols), dtype=np.uint8)
        for v in range(0, image_height, 2):
            for u in range(0, image_width, 2):
                d = depthFrame[v, u].astype(np.float32)
                if d < 100 or d > 5000:
                    continue
                Z = d / 1000.0
                X = (u - cx) * Z / fx
                r, c = world_to_grid(X, Z)
                if 0 <= r < rows and 0 <= c < cols:
                    hit_count[r, c] += 1

        occu_grid = (hit_count > 3).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        occu_grid = cv2.dilate(occu_grid, kernel, iterations=1)

        path = bfs(1 - occu_grid, start_pos, goal_pos)

        grid_vis = np.zeros((rows, cols, 3), dtype=np.uint8)
        grid_vis[occu_grid == 1] = [0, 0, 0]
        grid_vis[occu_grid == 0] = [255, 255, 255]
        for r, c in path:
            grid_vis[r, c] = [0, 255, 0]

        grid_vis = cv2.resize(grid_vis, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("3D World-Aligned Pathfinding Grid", grid_vis)
        cv2.imshow("RGB Camera", rgbFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
