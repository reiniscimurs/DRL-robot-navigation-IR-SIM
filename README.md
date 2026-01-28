[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/reiniscimurs/DRL-robot-navigation-IR-SIM)

**DRL Robot navigation in IR-SIM**

Deep Reinforcement Learning algorithm implementation for simulated robot navigation in IR-SIM. Using 2D laser sensor data
and information about the goal point a robot learns to navigate to a specified point in the environment.

![Example](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM/blob/master/out.gif)

**Installation**

* Package versioning is managed with poetry \
`pip install poetry`
* Clone the repository \
`git clone https://github.com/reiniscimurs/DRL-robot-navigation.git`
* Navigate to the cloned location and install using poetry \
`poetry install`

**Training the model**

* Run the training by executing the train.py file \
`poetry run python robot_nav/rl_train.py`

* To open tensorbord, in a new terminal execute \
`tensorboard --logdir runs`



**Sources**

| Package |                          Description                          |                              Source | 
|:--------|:-------------------------------------------------------------:|------------------------------------:| 
| IR-SIM  |                 Light-weight robot simulator                  | https://github.com/hanruihua/ir-sim |
| PythonRobotics  | Python code collection of robotics algorithms (Path planning) | https://github.com/AtsushiSakai/PythonRobotics |


**Models**

| Model            |                                           Description                                           |                    Model                           Source | 
|:-----------------|:-----------------------------------------------------------------------------------------------:|----------------------------------------------------------:|
| TD3              |                      Twin Delayed Deep Deterministic Policy Gradient model                      | https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2 | 
| SAC              |                                     Soft Actor-Critic model                                     |                https://github.com/denisyarats/pytorch_sac | 
| PPO              |                               Proximal Policy Optimization model                                |            https://github.com/nikhilbarhate99/PPO-PyTorch | 
| DDPG             |                            Deep Deterministic Policy Gradient model                             |                                          Updated from TD3 | 
| CNNTD3           |                          TD3 model with 1D CNN encoding of laser state                          |                                                         - |
| RCPG             | Recurrent Convolution Policy Gradient - adding recurrence layers (lstm/gru/rnn) to CNNTD3 model |                                                         - |
| MARL: TD3-G2ANet |                     G2ANet attention encoder for TD3 model in MARL setting                      |                                      G2ANet adapted from https://github.com/starry-sky6688/MARL-Algorithms |
| MARL: TD3-IGS    |                 In-Graph Softmax attention model for TD3 model in MARL setting                  |                                                         - |

**Max Upper Bound Models**

Models that support the additional loss of Q values exceeding the maximal possible Q value in the episode. Q values that exceed this upper bound are used to calculate a loss for the model. This helps to control the overestimation of Q values in off-policy actor-critic networks.
To enable max upper bound loss set `use_max_bound = True` when initializing a model.

| Model  |  
|:-------|
| TD3    | 
| DDPG   | 
| CNNTD3 |

**MARL Models**

Multi agent RL setting with training multiple robots a the same time with a single policy. Implementation does not use sensor informaition and only exchanged graph messages are used for navigation.

Video of results:

[![EXPERIMENTS](https://img.youtube.com/vi/SGl7sil_dpo/0.jpg)](https://www.youtube.com/watch?v=SGl7sil_dpo)
