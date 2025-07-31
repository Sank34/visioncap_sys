import depthai as dai
import numpy as np
import cv2
import math
import open3d as o3d
from collections import deque

# Constants
image_height = 400
image_width = 640
fx = 400
fy = 400
cx = image_width // 2
cy = image_height // 2

bev_size = 4.0
bev_resolution = 0.1
bev_cells = int(bev_size / bev_resolution)
bev_map = np.zeros((bev_cells, bev_cells), dtype=np.uint8)

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
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 1:
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

def find_valid_point(binary_map, center, radius=5):
    h, w = binary_map.shape
    cy, cx = center
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and binary_map[ny, nx] == 1:
                return (ny, nx)
    return None

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

        points = []
        colors = []
        cloud_vis = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for v in range(0, image_height, 2):
            for u in range(0, image_width, 2):
                d = depthFrame[v, u].astype(np.float32)
                if d < 100 or d > 5000:
                    continue
                Z = d / 1000.0
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                color = rgbFrame[v, u] / 255.0
                points.append((X, Y, Z))
                colors.append(color)
                cv2.circle(cloud_vis, (u, v), 1, (255, 255, 255), -1)

        bev_map.fill(0)
        for X, Y, Z in points:
            bev_x = int((X + bev_size / 2) / bev_resolution)
            bev_y = int((Z) / bev_resolution)
            if 0 <= bev_x < bev_cells and 0 <= bev_y < bev_cells:
                bev_map[bev_cells - bev_y - 1, bev_x] = 255

        binary_map = (bev_map == 0).astype(np.uint8)
        start = (bev_cells - 1, bev_cells // 2)
        goal = (0, bev_cells // 2)

        path_bev = []
        if binary_map[start[0], start[1]] != 1 or binary_map[goal[0], goal[1]] != 1:
            print("[WARN] BEV start or goal is blocked.")
        else:
            path_bev = bfs(binary_map, start, goal)

        bev_color = cv2.cvtColor(bev_map, cv2.COLOR_GRAY2BGR)
        for r, c in path_bev:
            y = int((bev_cells - r - 1) * image_height / bev_cells)
            x = int(c * image_width / bev_cells)
            cv2.circle(bev_color, (x, y), 2, (0, 255, 0), -1)

        # Depth map-based pathfinding
        depth_mask = (depthFrame > 800) & (depthFrame < 3000)
        depth_binary = depth_mask.astype(np.uint8)

        depth_start = find_valid_point(depth_binary, (image_height - 1, image_width // 2))
        depth_goal = find_valid_point(depth_binary, (0, image_width // 2))

        path_depth = []
        if not depth_start or not depth_goal:
            print("[WARN] Could not find valid start or goal nearby.")
        else:
            path_depth = bfs(depth_binary, depth_start, depth_goal)
            print(f"[INFO] Depth path length: {len(path_depth)}")

        depth_color = cv2.normalize(depthFrame, None, 100, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_GRAY2BGR)
        for r, c in path_depth:
            cv2.circle(depth_color, (c, r), 2, (0, 0, 255), -1)

        cv2.imshow("Bird's Eye View Occupancy Map", cv2.resize(bev_color, (640, 640), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("Depth Map + Path", depth_color)
        cv2.imshow("RGB Camera", rgbFrame)
        cv2.imshow("Point Cloud Preview (2D mask)", cloud_vis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("[INFO] Launching 3D colored point cloud viewer...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd], window_name='OAK-D Colored Point Cloud')

    cv2.destroyAllWindows()
