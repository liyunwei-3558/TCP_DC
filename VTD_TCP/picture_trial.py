'''
Author: Yunwei Li 1084087910@qq.com
Date: 2023-09-09 20:09:59
LastEditors: Yunwei Li 1084087910@qq.com
LastEditTime: 2023-09-09 22:08:17
FilePath: /TCP_DC/VTD_TCP/picture_trial.py
Description: 

Copyright (c) 2023 by Tsinghua University, All Rights Reserved. 
'''
"""
实现单图片网络推理
"""

import os
import json
import datetime
import pathlib
import time
import cv2
from collections import deque
import math
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TCP
from TCP.config import my_GlobalConfig
from team_code.planner import RoutePlanner

from VehicleControl import VehicleControl

SAVE_PATH = os.environ.get('SAVE_PATH', None)

class TCPAgent():
    def __init__(self, path_to_conf_file):
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = my_GlobalConfig()
        self.net = TCP(self.config)


        ckpt = torch.load(path_to_conf_file)
        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.","")
            new_state_dict[new_key] = value
        self.net.load_state_dict(new_state_dict, strict = False)
        self.net.cuda()
        self.net.eval()
        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        self.last_steers = deque()
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self.config.map_gps_mean) * self.config.map_gps_scale

        return gps    
    
    
    def tick(self, input_data):
        """ Process measurement data from the tick data. 
        And update the information of ego car, including next waypoint

        Args:
            input_data (dict): rgb, gps, speed, compass/imu, next_wp

        Returns:
            _type_: _description_
        """
        
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        # bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]  # TODO: Only use theta?
        
        
        # TODO:Robustness check
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                # 'bev': bev
                }

        pos = self._get_position(result)
        result['gps'] = pos
        # next_wp, next_cmd = self._route_planner.run_step(pos)
        next_wp=input_data['next_wp']
        # result['next_command'] = next_cmd.value

    # TODO: This might be specifically fixed according to CARLA
        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    def _init(self):
        
        self.initialized = True

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data) # get tick_data
        if self.step < self.config.seq_len:
            rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)
#TODO: change control cmd
            control = VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            
            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        
        # TODO： what's the usage of 'next_command'?
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
        speed = speed / 12
        rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                        torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        state = torch.cat([speed, target_point, cmd_one_hot], 1)

        pred= self.net(rgb, state, target_point)

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        self.pid_metadata = metadata_traj
        control = VehicleControl()

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
            control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
            control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
            control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
            control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if control.brake > 0.5:
            control.throttle = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(control.steer)))
        #chech whether ego is turning
        # num of steers larger than 0.1
        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1

        else:
            self.status = 0

        self.pid_metadata['status'] = self.status

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        return control