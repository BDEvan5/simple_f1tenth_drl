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
    
NUMBER_SCANS = 2
NUMBER_BEAMS = 20
MAX_SPEED = 4
MAX_STEER = 0.4
RANGE_FINDER_SCALE = 10

class EndToEndTrain: 
    def __init__(self, map_name, agent_name, Agent):
        self.name = agent_name
        self.path = "Data/" + agent_name + "/"
        init_file_struct(self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None
        self.std_track = TrackLine(map_name, False)
        self.reward_generator = ProgressReward(self.std_track)

        self.scan_buffer = np.zeros((NUMBER_SCANS, NUMBER_BEAMS))
        self.state_space = NUMBER_SCANS * NUMBER_BEAMS
        self.action_space = 2

        self.t_his = TrainHistory(self.path)
        self.agent = Agent(self.state_space, self.action_space)

    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        self.add_memory_entry(obs, nn_state)
        self.state = obs
            
        if obs['linear_vels_x'][0] < 1: # prevents unstable behavior at low speeds
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
    
    
    def __init__(self, Agent):
        self.scan_buffer = np.zeros((NUMBER_SCANS, NUMBER_BEAMS))
        self.state_space = NUMBER_SCANS * NUMBER_BEAMS
        self.action_space = 2
        
        self.agent = Agent(self.state_space, self.action_space)
        

    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        if obs['linear_vels_x'][0] < 1: # prevents unstable behavior at low speeds
            action = np.array([0, 2])
            return action

        nn_act = self.agent.act(nn_state)
        action = self.transform_action(nn_act)
        
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