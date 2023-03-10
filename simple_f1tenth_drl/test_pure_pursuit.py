from simple_f1tenth_drl.f1tenth_gym.f110_env import F110Env
from simple_f1tenth_drl.Planners.PurePursuit import PurePursuit


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
        

def test_pure_pursuit():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    n_test_laps = 5
    test_name = "my_best_pure_pursuit"
    
    env = F110Env(map=map_name, num_agents=1)
    planner = PurePursuit(map_name, test_name)
    
    run_simulation_loop_laps(env, planner, n_test_laps, 1)
  
def test_pure_pursuit_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    n_test_laps = 2
    
    for map_name in map_names:
        test_name = f"PurePursuit_{map_name}"
        
        env = F110Env(map=map_name, num_agents=1)
        planner = PurePursuit(map_name, test_name)
        
        run_simulation_loop_laps(env, planner, n_test_laps, 1)
  
  
if __name__ == "__main__":
    # test_pure_pursuit()
    test_pure_pursuit_all_maps()