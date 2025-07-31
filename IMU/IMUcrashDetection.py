import depthai as dai
import math

pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("imu")

imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xout.input)

CRASH_THRESHOLD = 22  #  m/s^2 (aprox 2.5g)

with dai.Device(pipeline) as device:
    imuQueue = device.getOutputQueue(name="imu", maxSize=10, blocking=False)

    while True:
        imuData = imuQueue.get()
        for packet in imuData.packets:
            acc = packet.acceleroMeter
            ax, ay, az = acc.x, acc.y, acc.z

            # compute acceleration magnitude
            a_mag = math.sqrt(ax**2 + ay**2 + az**2)

            print(f"Accel: {a_mag:.2f} m/s²")

            if a_mag > CRASH_THRESHOLD:
                print("CRASH or IMPACT DETECTED!")