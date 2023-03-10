import numpy as np
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine

import os, shutil
from simple_f1tenth_drl.PlannerUtils.VehicleStateHistory import VehicleStateHistory

NUMBER_SCANS = 2
NUMBER_BEAMS = 20
MAX_SPEED = 4
MAX_STEER = 0.4
RANGE_FINDER_SCALE = 10

 

class EndToEndTest: 
    def __init__(self, agent, map_name, test_name):
        self.scan_buffer = np.zeros((NUMBER_SCANS, NUMBER_BEAMS))
        self.state_space = NUMBER_SCANS * NUMBER_BEAMS
        self.action_space = 2
        
        self.agent = agent
        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)
        self.track_line = TrackLine(map_name, False, False)
        

    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        if obs['linear_vels_x'][0] < 1: # prevents unstable behavior at low speeds
            action = np.array([0, 2])
            return action

        nn_act = self.agent.act(nn_state)
        action = self.transform_action(nn_act)
        
        self.vehicle_state_history.add_memory_entry(obs, action)
        
        return action 

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scans'][0]) 

        scaled_scan = scan/RANGE_FINDER_SCALE
        scan = np.clip(scaled_scan, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(NUMBER_SCANS):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (NUMBER_BEAMS * NUMBER_SCANS))

        return nn_obs
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] *   MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])

        return action
    
    
    def done_callback(self, final_obs):
        self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Test lap complete --> Time: {final_obs['lap_times'][0]:.2f}, Colission: {bool(final_obs['collisions'][0])}, Lap p: {progress:.1f}%")
