'''
Author: Yunwei Li 1084087910@qq.com
Date: 2023-09-09 21:45:58
LastEditors: Yunwei Li 1084087910@qq.com
LastEditTime: 2023-09-09 21:47:04
FilePath: /TCP_DC/VTD_TCP/VehicleControl.py
Description: 

Copyright (c) 2023 by Tsinghua University, All Rights Reserved. 
'''


class VehicleControl:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0