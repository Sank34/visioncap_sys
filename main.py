import numpy as np
import matplotlib.pyplot as plt

# Simulated constants
image_height = 400
image_width = 640
grid_rows = 20
grid_cols = 20
cell_height = image_height // grid_rows
cell_width = image_width // grid_cols

fy = 400
cy = image_height // 2

# depth frame (1000mm to 3000mm)
np.random.seed(42)
low_b = 500
high_b = 1000
depth_frame = np.random.randint(low_b, high_b, size=(image_height, image_width))

# pitch angle from IMU
pitch_deg = -5  # degrees
pitch_rad = np.radians(pitch_deg)

N = 20  # rows from the bottom
bottom_strip = depth_frame[-N:, :]
valid = bottom_strip[(bottom_strip > 100) & (bottom_strip < 5000)]
ground_depth = np.median(valid)

camera_height_mm = ground_depth * np.abs(np.sin(pitch_rad))

occupancy_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
height_map = np.zeros((grid_rows, grid_cols))
height_threshold = 200  # mm (20 cm obstacle threshold)

for i in range(grid_rows):
    for j in range(grid_cols):
        cell = depth_frame[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
        valid_depths = cell[(cell > 100) & (cell < 5000)]
        if len(valid_depths) == 0:
            occupancy_grid[i, j] = 0
            height_map[i, j] = 0
            continue

        cell_depth = np.median(valid_depths)
        v = i * cell_height + cell_height // 2
        theta_p = np.arctan2((v - cy), fy)

        object_height = camera_height_mm - cell_depth * np.sin(pitch_rad + theta_p)

        height_map[i, j] = object_height
        occupancy_grid[i, j] = 1 if object_height > height_threshold else 0

plt.figure(figsize=(8, 6))
plt.imshow(occupancy_grid, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
plt.title("Updated 2.5D Occupancy Grid (-5 Pitch)")
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.colorbar(ticks=[0, 1], label='Occupancy')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(height_map, cmap='viridis')
plt.title("Estimated Object Height Map (in mm)")
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.colorbar(label='Height (mm)')
plt.show()

unique_vals, counts = np.unique(occupancy_grid, return_counts=True)
print("Occupancy Grid Breakdown:", dict(zip(unique_vals, counts)))