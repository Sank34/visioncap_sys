import depthai as dai
import math
import time

pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("imu")

imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xout.input)

CRASH_THRESHOLD = 16  # m/s^2
COOLDOWN_SECONDS = 2.0  # repeat triggers

last_crash_time = 0

with dai.Device(pipeline) as device:
    imuQueue = device.getOutputQueue(name="imu", maxSize=10, blocking=False)
    ok = True
    while ok:
        imuData = imuQueue.get()
        for packet in imuData.packets:
            acc = packet.acceleroMeter
            ax, ay, az = acc.x, acc.y, acc.z

            a_mag = math.sqrt(ax**2 + ay**2 + az**2) #calc acc magnitude

            print(f"Acceleration: {a_mag:.2f} m/s²")

            current_time = time.time()
            if a_mag > CRASH_THRESHOLD and (current_time - last_crash_time) > COOLDOWN_SECONDS:
                last_crash_time = current_time
                timestamp = time.strftime('%H:%M:%S', time.localtime(current_time))
                print(f" CRASH DETECTED at {timestamp}!")
                ok = False
                break
