from simple_f1tenth_drl.f1tenth_gym.f110_env import F110Env
from simple_f1tenth_drl.LearningAlgorithms.sac import TrainSAC, TestSAC
from simple_f1tenth_drl.LearningAlgorithms.ddpg import TrainDDPG, TestDDPG
from simple_f1tenth_drl.LearningAlgorithms.td3 import TrainTD3, TestTD3

from simple_f1tenth_drl.Planners.TrainTrajectoryFollow import TrajectoryFollowTrain
from simple_f1tenth_drl.Planners.TestTrajectoryFollow import TrajectoryFollowTest
from simple_f1tenth_drl.Planners.TestEndToEnd import EndToEndTest
from simple_f1tenth_drl.Planners.TrainEndToEnd import EndToEndTrain

import numpy as np

RENDER_ENV = False
# RENDER_ENV = True


def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = n_sim_steps
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1

            if RENDER_ENV: env.render('human')
            
        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))   
        

def test_endToEnd_agent():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    agent_name = "myFavouriteAgent_SAC"
    n_test_laps = 5
    
    env = F110Env(map=map_name, num_agents=1)
    
    TestAgent = TestSAC(agent_name, f"Data/{agent_name}/") # or DDPG, TD3
    planner = EndToEndTest(TestAgent, map_name, agent_name)
    run_simulation_loop_laps(env, planner, n_test_laps)
  
  
def test_trajectoryFollowing_agent():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    agent_name = "myFavouriteAgentTF_SAC"
    n_test_laps = 5
    
    env = F110Env(map=map_name, num_agents=1)

    
    test_agent = TestSAC(agent_name, f"Data/{agent_name}/") # or DDPG, TD3
    planner = TrajectoryFollowTest(test_agent, map_name, agent_name)
    run_simulation_loop_laps(env, planner, n_test_laps)
  
def test_endToEnd_agent():
    map_names = ["aut", "esp", "gbr", "mco"]
    agent_name = "myFavouriteAgent_SAC"
    n_test_laps = 2
    
    for map_name in map_names:
        env = F110Env(map=map_name, num_agents=1)
        TestAgent = TestSAC(agent_name, f"Data/{agent_name}/") # or DDPG, TD3
        planner = EndToEndTest(TestAgent, map_name, agent_name)
        run_simulation_loop_laps(env, planner, n_test_laps)
  
  
def test_trajectoryFollowing_agent():
    map_names = ["aut", "esp", "gbr", "mco"]
    agent_name = "myFavouriteAgentTF_SAC"
    n_test_laps = 2
    
    for map_name in map_names:
        env = F110Env(map=map_name, num_agents=1)
        test_agent = TestSAC(agent_name, f"Data/{agent_name}/") # or DDPG, TD3
        planner = TrajectoryFollowTest(test_agent, map_name, agent_name)
        run_simulation_loop_laps(env, planner, n_test_laps)
  
  
if __name__ == "__main__":
    # test_endToEnd_agent()
    test_trajectoryFollowing_agent()
    