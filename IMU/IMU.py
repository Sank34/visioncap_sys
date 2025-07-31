import depthai as dai
import cv2
import time as tt
import math
'''
MAIN FUNCTIONS 
pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)

imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)

imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
'''

pipeline = dai.Pipeline()

imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW,400)
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW,500)

imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

imu.out.link(xlinkOut.input)

with dai.Device(pipeline) as device:

    def timeDeltaToMS(delta) -> float: # calc ms
        return delta.total_seconds() * 1000
    imuQueue = device.getOutputQueue(name = "imu", maxSize = 50, blocking = False)
    baseTs = None
    while True:
        imuData = imuQueue.get() # wait for new data
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

            imuF = "{:.06f}"
            tsF = "{:.06f}"

            print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
            print(
                f"Accelerometer [m/s^2]: x: {imuF.format(accVal.x)} y: {imuF.format(accVal.y)} z: {imuF.format(accVal.z)}")
            print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            print(
                f"Gyroscope [rad/s]: x: {imuF.format(gyroVal.x)} y: {imuF.format(gyroVal.y)} z: {imuF.format(gyroVal.z)} ")

            if cv2.waitKey(1) == ord('q'):
                break
