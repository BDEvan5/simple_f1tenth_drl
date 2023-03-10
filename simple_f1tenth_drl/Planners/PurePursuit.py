"""
Partial code source: https://github.com/f1tenth/f1tenth_gym
Example waypoint_follow.py from f1tenth_gym
Specific function used:
- nearest_point_on_trajectory_py2
- first_point_on_trajectory_intersecting_circle
- get_actuation

Adjustments have been made

"""

import numpy as np
from numba import njit
import os
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine
from simple_f1tenth_drl.PlannerUtils.VehicleStateHistory import VehicleStateHistory

LOOKAHEAD_DISTANCE = 1
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8


class PurePursuit:
    def __init__(self, map_name, test_name):
        path = f"Data/" + test_name + "/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        self.track_line = TrackLine(map_name, True, False)

        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)

        self.counter = 0

    def plan(self, obs):
        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        theta = obs['poses_theta'][0]
        
        lookahead_point = self.track_line.get_lookahead_point(position, LOOKAHEAD_DISTANCE)

        if obs['linear_vels_x'][0] < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(theta, lookahead_point, position, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            
        speed = min(speed_raceline, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])
        
        self.vehicle_state_history.add_memory_entry(obs, action)

        return action

    def done_callback(self, final_obs):
        self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Test lap complete --> Time: {final_obs['lap_times'][0]:.2f}, Colission: {bool(final_obs['collisions'][0])}, Lap p: {progress:.1f}%")



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

