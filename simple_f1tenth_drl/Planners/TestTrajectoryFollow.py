import numpy as np
from simple_f1tenth_drl.PlannerUtils.TrainHistory import TrainHistory
from simple_f1tenth_drl.PlannerUtils.RewardSignals import ProgressReward
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine
from simple_f1tenth_drl.PlannerUtils.VehicleStateHistory import VehicleStateHistory

    
NUMBER_WAYPOINTS = 10
MAX_SPEED = 4
MAX_STEER = 0.4
RANGE_FINDER_SCALE = 10

 

class TrajectoryFollowTest: 
    def __init__(self, agent, map_name, agent_name):
        self.agent = agent
        self.racing_line = TrackLine(map_name, True)
        self.std_track = TrackLine(map_name, False)
        
        self.vehicle_state_history = VehicleStateHistory(agent_name, map_name)
        

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
        idx, dists = self.racing_line.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        speed = obs['linear_vels_x'][0]
        anglular_vel = obs['ang_vels_z'][0]
        steering_angle = obs['steering_deltas'][0]
        
        upcomings_inds = np.arange(idx, idx+NUMBER_WAYPOINTS)
        if idx + NUMBER_WAYPOINTS >= self.racing_line.N:
            n_start_pts = idx + NUMBER_WAYPOINTS - self.racing_line.N
            upcomings_inds[NUMBER_WAYPOINTS - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.racing_line.wpts[upcomings_inds]
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        
        speeds = self.racing_line.vs[upcomings_inds]
        scaled_speeds = np.clip(speeds / MAX_SPEED, 0, 1)
        
        relative_pts = np.concatenate((relative_pts, scaled_speeds[:, None]), axis=-1)
        state = np.concatenate((relative_pts.flatten(), np.array([speed, anglular_vel, steering_angle])))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] *   MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])

        return action
    
    def done_callback(self, final_obs):
        # self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        # self.vehicle_state_history.save_history()
        
        progress = self.std_track.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Test lap complete --> Time: {final_obs['lap_times'][0]:.2f}, Colission: {bool(final_obs['collisions'][0])}, Lap p: {progress:.1f}%")
    
 
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts



