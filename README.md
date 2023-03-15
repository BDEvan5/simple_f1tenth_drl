# simple_f1tenth_drl
This is a repo with example implementations of deep reinforcement learning algorithms for F1tenth racing.
The aim is to provide a simple implementation that can help people to understand how to use DRL algorithms for autonomous racing.


# Use

## Installation

```
git clone https://github.com/BDEvan5/simple_f1tenth_drl
cd simple_f1tenth_drl
mkdir Data/
pip install -e .
```

## Task Description

The `train_agents.py` file is the main entry point to train agents.
Agents can be trained for two tasks:
1. **Trajectory Following:** 10 upcoming waypoints (relative positions and speed), and the angluar speed, linear speed and steering angle are given to the agent.
2. **End-to-end:** The agent receives 20 beams from the LiDAR scan for the previous and current vehicle positions.

In both cases, the action space is the steering angle and vehicle speed.

The `test_agents.py` file has functions to test the agents (for each task) on the training maps, or on all of the maps.

The pure pursuit planner is given for comparison and can be tested using the `test_pure_pursuit.py` file.

## Component Description

### f1tenth_gym

This is the simulator from www.github.com/f1tenth/f1tenth_gym. It contains minor modifications, e.g. only provides 20 beams, but remains largely the same.

### Learning Algorithms

The DDPG, TD3 and SAC learning algorithms from the repo, github.com/BDEvan5/BaselineDRL.
They are simple, single file implementations for use in robotic control.
They use neural networks in the `PlannerUtils/Networks.py` file and replay buffers from `PlannerUtils/ReplayBuffers.py`. 
Each algorith contains an act and a train method.

### Data Saving

**Training Agents:**

For each agent, the a folder is created inside the `Data/` directory with the agent's name.
For the training, the model parameters of the actor are saved as a .pth file so that the model can be loaded for testing.
The .pth file is a sterilised verion of the actor neural network.

The `TrainHistory` class is used to store the rewards earned and average progress during training.
Durign and after the training, plots of the rewards earned per lap and the average progress during training are saved to the agent's directory.

**Testing:**

For the tests, a subfolder is created called, `TestingMAPNAME`, with the test map name in the foldername.
Inside, the full vehicle state and actions selected at each timestep is saved for analysis.

The DataTools can analyse this data and make pretty plots.
- GenerateTrajectoryAnalysis: for each vehicle and test, plots the speed, slip, centre line deviation etc, for the trajectory around the traick.
- GenerateVelocityProfiles: plots a colour line to represent the velocity profile around the race track.


## Improvements:
- Add experiments to compare algorithms 
- Add experiment to compare path following vs end-to-end
- Refactor pure pursuit to use a single folder with multiple Testing folders for different maps.






