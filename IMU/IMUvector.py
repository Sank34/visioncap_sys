#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math

device = dai.Device()

imuType = device.getConnectedIMU()
imuFirmwareVersion = device.getIMUFirmwareVersion()
print(f"IMU type: {imuType}, firmware version: {imuFirmwareVersion}")
