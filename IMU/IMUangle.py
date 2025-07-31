import depthai as dai
import cv2
import time as tt
import math

pipeline = dai.Pipeline()

imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)

imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

imu.out.link(xlinkOut.input)

with dai.Device(pipeline) as device:

    def timeDeltaToMS(delta) -> float:
        return delta.total_seconds() * 1000

    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    baseTs = None

    while True:
        imuData = imuQueue.get()
        imuPackets = imuData.packets

        for imuPacket in imuPackets:
            accVal = imuPacket.acceleroMeter
            gyroVal = imuPacket.gyroscope

            acceleroTs = accVal.getTimestampDevice()
            gyroTs = gyroVal.getTimestampDevice()

            if baseTs is None:
                baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs

            acceleroTs = timeDeltaToMS(acceleroTs - baseTs)
            gyroTs = timeDeltaToMS(gyroTs - baseTs)

            ax, ay, az = accVal.x, accVal.y, accVal.z

            # PITCH CALCULATION (in degrees)
            pitch_rad = math.atan2(-ax, math.sqrt(ay**2 + az**2))
            roll_rad = math.atan2(ay,az)
            pitch_deg = math.degrees(pitch_rad)
            roll_deg = math.degrees(roll_rad)

            # Output
            # print(f"Accelerometer timestamp: {acceleroTs:.3f} ms")
            # print(f"Accelerometer [m/s^2]: x: {ax:.6f} y: {ay:.6f} z: {az:.6f}")
            # print(f"Gyroscope timestamp: {gyroTs:.3f} ms")
            # print(f"Gyroscope [rad/s]: x: {gyroVal.x:.6f} y: {gyroVal.y:.6f} z: {gyroVal.z:.6f}")
            # print(f"Estimated Pitch: {pitch_deg:.2f}°\n")
            # print(f"Estimated Roll: {roll_deg:.2f}°\n")

            x = math.degrees(gyroVal.x)
            y = math.degrees(gyroVal.y)
            z = math.degrees(gyroVal.z)

            print(f"Angle x: {x:.2f} deg")
            print(f"Angle y: {y:.2f} deg")
            print(f"Angle z: {z:.2f} deg")

            if cv2.waitKey(1) == ord('q'):
                break