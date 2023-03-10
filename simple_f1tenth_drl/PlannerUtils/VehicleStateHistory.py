import numpy as np
import os

class VehicleStateHistory:
    def __init__(self, test_name, map_name):
        self.vehicle_name = test_name
        self.path = "Data/" + test_name + f"/Testing_{map_name.upper()}" 
        if os.path.exists(self.path) == False:
            os.mkdir(self.path)
        self.states = []
        self.actions = []
    
        self.lap_n = 0
    
    def add_memory_entry(self, obs, action):
        state = obs['full_states'][0]

        self.states.append(state)
        self.actions.append(action)
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)
        
        np.save(self.path + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy", lap_history)

        self.states = []
        self.actions = []
        self.lap_n += 1
        
    


if __name__ == "__main__":
    pass