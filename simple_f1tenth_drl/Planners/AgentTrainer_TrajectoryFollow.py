import numpy as np
from simple_f1tenth_drl.PlannerUtils.TrainHistory import TrainHistory
from simple_f1tenth_drl.PlannerUtils.RewardSignals import ProgressReward
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine

import os, shutil

def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)
    

class AgentTrainer_TrajectoryFollow: 
    def __init__(self, map_name, agent_name, agent):
        self.name = agent_name
        self.path = "Data/" + agent_name + "/"
        init_file_struct(self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None
        self.std_track = TrackLine(map_name, False)
        self.reward_generator = ProgressReward(self.std_track)

        self.agent = agent
        self.t_his = TrainHistory(self.path)

    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        self.add_memory_entry(obs, nn_state)
        self.state = obs
            
        if obs['linear_vels_x'][0] < self.v_min_plan:
            self.action = np.array([0, 2])
            return self.action

        self.nn_state = nn_state 
        self.nn_act = self.agent.act(self.nn_state)
        self.action = self.transform_action(self.nn_act)
        
        self.agent.train()

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            reward = self.reward_generator(s_prime, self.state)
            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_callback(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.reward_generator(s_prime, self.state)
        progress = self.std_track.calculate_progress_percent([s_prime['poses_x'][0], s_prime['poses_y'][0]]) * 100
        
        self.t_his.lap_done(reward, progress, False)
        print(f"Episode: {self.t_his.ptr}, Step: {self.t_his.t_counter}, Lap p: {progress:.1f}%, Reward: {self.t_his.rewards[self.t_his.ptr-1]:.2f}")

        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)
        self.nn_state = None
        self.state = None

        self.save_training_data()

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.name, self.path)
        
    def transform_obs(self, obs):
        idx, dists = self.track.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        speed = obs['linear_vels_x'][0]
        anglular_vel = obs['ang_vels_z'][0]
        steering_angle = obs['steering_deltas'][0]
        
        upcomings_inds = np.arange(idx, idx+self.n_wpts)
        if idx + self.n_wpts >= self.track.N:
            n_start_pts = idx + self.n_wpts - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        
        speeds = self.track.vs[upcomings_inds]
        scaled_speeds = np.clip(speeds / self.max_speed, 0, 1)
        
        relative_pts = np.concatenate((relative_pts, scaled_speeds[:, None]), axis=-1)
        state = np.concatenate((relative_pts.flatten(), np.array([speed, anglular_vel, steering_angle])))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action
    
 
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
    