import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
image_height = 400
image_width = 640
grid_rows = 40
grid_cols = 40
cell_height = image_height // grid_rows
cell_width = image_width // grid_cols
fy = 400
cy = image_height // 2
height_threshold = 200
pitch_deg = -10
pitch_rad = np.radians(pitch_deg)

low_b = 500
high_b = 1000
base_depth = np.tile(np.linspace(600, 1200, image_height).reshape(-1, 1), (1, image_width))
noise = np.random.normal(loc=0, scale=50, size=(image_height, image_width))
depth_frame = base_depth + noise
depth_frame = np.clip(depth_frame, low_b, high_b).astype(np.uint16)

N = 20
bottom_strip = depth_frame[-N:, :]
valid = bottom_strip[(bottom_strip > 100) & (bottom_strip < 5000)]
ground_depth = np.median(valid)
camera_height_mm = ground_depth * np.abs(np.sin(pitch_rad))

occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
height_map = np.zeros((grid_rows, grid_cols))

for i in range(grid_rows):
    for j in range(grid_cols):
        cell = depth_frame[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
        valid_depths = cell[(cell > 100) & (cell < 5000)]
        if len(valid_depths) == 0:
            continue
        cell_depth = np.min(valid_depths)
        v = i * cell_height + cell_height // 2
        theta_p = np.arctan2((v - cy), fy)
        object_height = camera_height_mm - cell_depth * np.sin(pitch_rad + theta_p)
        object_height = max(0, object_height)
        height_map[i, j] = object_height
        occupancy_grid[i, j] = 1 if object_height > height_threshold else 0

plt.figure(figsize=(8, 6))
plt.imshow(occupancy_grid, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
plt.title("Occupancy Grid (Random Noise + Floor Gradient)")
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.colorbar(ticks=[0, 1], label='Occupancy')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(height_map, cmap='viridis')
plt.title("Object Height Map (Random Noise + Floor Gradient)")
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.colorbar(label='Height (mm)')
plt.show()

# arctan((v-cy) / fy)
# v -> pixel row of the grid,
# cy -> principal point y,
# fy -> focal length in pixels