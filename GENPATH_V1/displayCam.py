import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Create color camera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 400)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Create XLink output node for RGB preview
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start device
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    print("Showing RGB camera preview. Press 'q' to quit.")
    while True:
        in_rgb = rgb_queue.get()
        frame = in_rgb.getCvFrame()

        cv2.imshow("RGB Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()