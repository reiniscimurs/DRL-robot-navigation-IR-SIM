from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.utils import get_max_bound


class Actor(nn.Module):
    """
    Actor network for the CNNTD3 agent.

    This network takes as input a state composed of laser scan data, goal position encoding,
    and previous action. It processes the scan through a 1D CNN stack and embeds the other
    inputs before merging all features through fully connected layers to output a continuous
    action vector.

    Args:
        action_dim (int): The dimension of the action space.

    Architecture:
        - 1D CNN layers process the laser scan data.
        - Fully connected layers embed the goal vector (cos, sin, distance) and last action.
        - Combined features are passed through two fully connected layers with LeakyReLU.
        - Final action output is scaled with Tanh to bound the values.
    """

    def __init__(self, action_dim):
        super(Actor, self).__init__()

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.velocity_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.softsign = nn.Softsign()

    def forward(self, lidar, goal, velocity):
        """
        Forward pass through the Actor network.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
                              The last 5 elements are [distance, cos, sin, lin_vel, ang_vel].

        Returns:
            (torch.Tensor): Action tensor of shape (batch_size, action_dim),
                          with values in range [-1, 1] due to tanh activation.
        """
        if len(lidar.shape) == 1:
            lidar = lidar.unsqueeze(0)
            goal = goal.unsqueeze(0)
            velocity = velocity.unsqueeze(0)
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.velocity_embed(velocity))

        s = torch.concat((l, g, a), dim=-1)

        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.softsign(self.layer_3(s))
        return a

    def reward(self, batch_rewards, device):
        rewards = []
        for b in batch_rewards:
            rewards.append([b["collision"] + b["goal"] + b["step"]])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(-1,1)
        return rewards


class Critic(nn.Module):
    """
    Critic network for the CNNTD3 agent.

    The Critic estimates Q-values for state-action pairs using two separate sub-networks
    (Q1 and Q2), as required by the TD3 algorithm. Each sub-network uses a combination of
    CNN-extracted features, embedded goal and previous action features, and the current action.

    Args:
        action_dim (int): The dimension of the action space.

    Architecture:
        - Shared CNN layers process the laser scan input.
        - Goal and previous action are embedded and concatenated.
        - Each Q-network uses separate fully connected layers to produce scalar Q-values.
        - Both Q-networks receive the full state and current action.
        - Outputs two Q-value tensors (Q1, Q2) for TD3-style training and target smoothing.
    """

    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.velocity_embed = nn.Linear(2, 10)
        self.action_embed = nn.Linear(2, 10)


        self.layer_1 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, lidar, goal, velocity, action):
        """
        Forward pass through both Q-networks of the Critic.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
                              The last 5 elements are [distance, cos, sin, lin_vel, ang_vel].
            action (torch.Tensor): Current action tensor of shape (batch_size, action_dim).

        Returns:
            (tuple):
                - q1 (torch.Tensor): First Q-value estimate (batch_size, 1).
                - q2 (torch.Tensor): Second Q-value estimate (batch_size, 1).
        """
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))
        v = F.leaky_relu(self.velocity_embed(velocity))
        a = F.leaky_relu(self.action_embed(action))

        s = torch.concat((l, g, v, a), dim=-1)

        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

class Dev_Actor(nn.Module):
    """
    Actor network for the CNNTD3 agent.

    This network takes as input a state composed of laser scan data, goal position encoding,
    and previous action. It processes the scan through a 1D CNN stack and embeds the other
    inputs before merging all features through fully connected layers to output a continuous
    action vector.

    Args:
        action_dim (int): The dimension of the action space.

    Architecture:
        - 1D CNN layers process the laser scan data.
        - Fully connected layers embed the goal vector (cos, sin, distance) and last action.
        - Combined features are passed through two fully connected layers with LeakyReLU.
        - Final action output is scaled with Tanh to bound the values.
    """

    def __init__(self, action_dim):
        super(Dev_Actor, self).__init__()

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.base_action_embed = nn.Linear(2, 10)
        self.velocity_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.softsign = nn.Softsign()

    def forward(self, lidar, velocity, base_action):
        """
        Forward pass through the Actor network.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
                              The last 5 elements are [distance, cos, sin, lin_vel, ang_vel].

        Returns:
            (torch.Tensor): Action tensor of shape (batch_size, action_dim),
                          with values in range [-1, 1] due to tanh activation.
        """
        if len(lidar.shape) == 1:
            lidar = lidar.unsqueeze(0)
        if len(velocity.shape) == 1:
            velocity = velocity.unsqueeze(0)
        if len(base_action.shape) == 1:
            base_action = base_action.unsqueeze(0)

        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        a = F.leaky_relu(self.base_action_embed(base_action))
        v = F.leaky_relu(self.velocity_embed(velocity))

        s = torch.concat((l, a, v), dim=-1)

        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.softsign(self.layer_3(s))
        return a*2

    def reward(self, batch_rewards, device):
        rewards = []
        for b in batch_rewards:
            rewards.append([b["collision"]])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(-1, 1)
        return rewards


class Dev_Critic(nn.Module):
    """
    Critic network for the CNNTD3 agent.

    The Critic estimates Q-values for state-action pairs using two separate sub-networks
    (Q1 and Q2), as required by the TD3 algorithm. Each sub-network uses a combination of
    CNN-extracted features, embedded goal and previous action features, and the current action.

    Args:
        action_dim (int): The dimension of the action space.

    Architecture:
        - Shared CNN layers process the laser scan input.
        - Goal and previous action are embedded and concatenated.
        - Each Q-network uses separate fully connected layers to produce scalar Q-values.
        - Both Q-networks receive the full state and current action.
        - Outputs two Q-value tensors (Q1, Q2) for TD3-style training and target smoothing.
    """

    def __init__(self, action_dim):
        super(Dev_Critic, self).__init__()
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.velocity_embed = nn.Linear(2, 10)
        self.base_action_embed = nn.Linear(2, 10)
        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(46, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, lidar, velocity, base_action, action):
        """
        Forward pass through both Q-networks of the Critic.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
                              The last 5 elements are [distance, cos, sin, lin_vel, ang_vel].
            action (torch.Tensor): Current action tensor of shape (batch_size, action_dim).

        Returns:
            (tuple):
                - q1 (torch.Tensor): First Q-value estimate (batch_size, 1).
                - q2 (torch.Tensor): Second Q-value estimate (batch_size, 1).
        """
        lidar = lidar.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(lidar))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)
        v = F.leaky_relu(self.velocity_embed(velocity))
        b = F.leaky_relu(self.base_action_embed(base_action))
        a = F.leaky_relu(self.action_embed(action))

        s = torch.concat((l, v, b, a), dim=-1)

        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# CNNTD3 network
class dualCNNTD3(object):
    """
    CNNTD3 (Twin Delayed Deep Deterministic Policy Gradient with CNN-based inputs) agent for
    continuous control tasks.

    This class encapsulates the full implementation of the TD3 algorithm using neural network
    architectures for the actor and critic, with optional bounding for critic outputs to
    regularize learning. The agent is designed to train in environments where sensor
    observations (e.g., LiDAR) are used for navigation tasks.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
        max_action (float): Maximum magnitude of the action.
        device (torch.device): Torch device to use (CPU or GPU).
        lr (float): Learning rate for both actor and critic optimizers.
        save_every (int): Save model every N training iterations (0 to disable).
        load_model (bool): Whether to load a pre-trained model at initialization.
        save_directory (Path): Path to the directory for saving model checkpoints.
        model_name (str): Base name for the saved model files.
        load_directory (Path): Path to load model checkpoints from (if `load_model=True`).
        use_max_bound (bool): Whether to apply maximum Q-value bounding during training.
        bound_weight (float): Weight for the bounding loss term in total loss.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/dev_rl/dualCNNTD3/checkpoint"),
        model_name="devCNNTD3",
        load_directory=Path("robot_nav/dev_rl/dualCNNTD3/checkpoint"),
        use_max_bound=False,
        bound_weight=0.25,
    ):
        # Initialize the Actor network
        self.device = device
        self.actor = Actor(action_dim).to(self.device)
        self.actor_target = Actor(action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        self.dev_actor = Dev_Actor(action_dim).to(self.device)
        self.dev_actor_target = Dev_Actor(action_dim).to(self.device)
        self.dev_actor_target.load_state_dict(self.dev_actor.state_dict())
        self.dev_actor_optimizer = torch.optim.Adam(params=self.dev_actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(action_dim).to(self.device)
        self.critic_target = Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        self.dev_critic = Dev_Critic(action_dim).to(self.device)
        self.dev_critic_target = Dev_Critic(action_dim).to(self.device)
        self.dev_critic_target.load_state_dict(self.dev_critic.state_dict())
        self.dev_critic_optimizer = torch.optim.Adam(params=self.dev_critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.use_max_bound = use_max_bound
        self.bound_weight = bound_weight

    def get_action(self, obs, add_noise):
        """
        Selects an action for a given observation.

        Args:
            obs (np.ndarray): The current observation/state.
            add_noise (bool): Whether to add exploration noise to the action.

        Returns:
            (np.ndarray): The selected action.
        """
        if add_noise:
            action, dev_action = self.act(obs)
            action = (action + np.random.normal(0, 0.2, size=self.action_dim)).clip(-self.max_action, self.max_action)
            dev_action = (dev_action + np.random.normal(0, 0.2, size=self.action_dim)).clip(-self.max_action*2, self.max_action*2)
            return action, dev_action
        else:
            return self.act(obs)

    def act(self, state):
        """
        Computes the deterministic action from the actor network for a given state.

        Args:
            state (np.ndarray): Input state.

        Returns:
            (np.ndarray): Action predicted by the actor network.
        """
        # Function to get the action from the actor
        lidar = torch.tensor(state["lidar"], dtype=torch.float32, device=self.device)
        goal = torch.tensor(state["goal"],dtype=torch.float32, device=self.device)
        velocity = torch.tensor(state["velocity"], dtype=torch.float32, device=self.device)
        action = self.actor(lidar, goal, velocity)
        dev_action = self.dev_actor(lidar, action, velocity).cpu().data.numpy().flatten()
        return action.cpu().data.numpy().flatten(), dev_action

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """
        Trains the CNNTD3 agent using sampled batches from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Buffer storing environment transitions.
            iterations (int): Number of training iterations.
            batch_size (int): Size of each training batch.
            discount (float): Discount factor for future rewards.
            tau (float): Soft update rate for target networks.
            policy_noise (float): Std. dev. of noise added to target policy.
            noise_clip (float): Maximum value for target policy noise.
            policy_freq (int): Frequency of actor and target network updates.
            max_lin_vel (float): Maximum linear velocity for bounding calculations.
            max_ang_vel (float): Maximum angular velocity for bounding calculations.
            goal_reward (float): Reward value for reaching the goal.
            distance_norm (float): Normalization factor for distance in bounding.
            time_step (float): Time delta between steps.
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            lidar = []
            goal = []
            velocity = []
            for b in batch_states:
                lidar.append(b["lidar"])
                goal.append(b["goal"])
                velocity.append(b["velocity"])

            lidar = torch.tensor(lidar, dtype=torch.float32, device=self.device)
            goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self.device)

            next_lidar = []
            next_goal = []
            next_velocity = []
            for b in batch_next_states:
                next_lidar.append(b["lidar"])
                next_goal.append(b["goal"])
                next_velocity.append(b["velocity"])

            next_lidar = torch.tensor(next_lidar, dtype=torch.float32, device=self.device)
            next_goal = torch.tensor(next_goal, dtype=torch.float32, device=self.device)
            next_velocity = torch.tensor(next_velocity, dtype=torch.float32, device=self.device)

            action = torch.tensor(batch_actions, dtype=torch.float32, device=self.device)
            # reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
            done = torch.tensor(batch_dones, dtype=torch.float32, device=self.device).reshape(-1, 1)

            dev_action = action[...,2:]
            base_action = action[...,:2]
            executed_action = base_action + dev_action
            #------------------------------
            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_lidar, next_goal, next_velocity)
            with torch.no_grad():
                next_override = self.dev_actor_target(next_lidar, next_velocity, next_action)

            # Add noise to the action
            noise = torch.randn_like(next_action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_full_action = (next_action + next_override + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_lidar, next_goal, next_velocity, next_full_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation

            with torch.no_grad():
                deviation_loss = torch.linalg.norm(
                    dev_action,
                    dim=-1,
                    keepdim=True
                )

                target_Q = self.actor.reward(batch_rewards, self.device) - deviation_loss + (
                            (1 - done) * discount * target_Q)

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(lidar, goal, velocity, executed_action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            # ------------------------------




            deviation_loss = torch.linalg.norm(dev_action, dim=-1, keepdim=True)
            dev_next_action = self.dev_actor_target(next_lidar, next_velocity, next_action.detach())
            noise = torch.randn_like(dev_next_action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            dev_next_action = (dev_next_action + noise).clamp(-self.max_action*2, self.max_action*2)

            dev_target_Q1, dev_target_Q2 = self.dev_critic_target(next_lidar, next_velocity, next_action.detach(), dev_next_action)
            dev_target_Q = torch.min(dev_target_Q1, dev_target_Q2)
            with torch.no_grad():
                dev_target_Q = self.dev_actor.reward(batch_rewards, self.device) - deviation_loss + ((1 - done) * discount * dev_target_Q)
            dev_current_Q1, dev_current_Q2 = self.dev_critic(lidar, velocity, base_action, dev_action)

            dev_critic_loss = F.mse_loss(dev_current_Q1, dev_target_Q) + F.mse_loss(dev_current_Q2, dev_target_Q)

            self.dev_critic_optimizer.zero_grad()
            dev_critic_loss.backward()
            self.dev_critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                self.critic.eval()
                self.dev_critic.eval()
                actor_grad, _ = self.critic(lidar, goal, velocity, self.actor(lidar, goal, velocity))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                dev_actor_grad, _ = self.dev_critic(lidar, velocity, base_action, self.dev_actor(lidar, velocity, base_action))
                dev_actor_grad = -dev_actor_grad.mean()
                self.dev_actor_optimizer.zero_grad()
                dev_actor_grad.backward()
                self.dev_actor_optimizer.step()

                self.critic.train()
                self.dev_critic.train()

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.dev_actor.parameters(), self.dev_actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.dev_critic.parameters(), self.dev_critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        """
        Saves the current model parameters to the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to save the model files.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(
            self.actor_target.state_dict(),
            "%s/%s_actor_target.pth" % (directory, filename),
        )
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(
            self.critic_target.state_dict(),
            "%s/%s_critic_target.pth" % (directory, filename),
        )

    def load(self, filename, directory):
        """
        Loads model parameters from the specified directory.

        Args:
            filename (str): Base filename for saved files.
            directory (Path): Path to load the model files from.
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.actor_target.load_state_dict(
            torch.load("%s/%s_actor_target.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.critic_target.load_state_dict(
            torch.load("%s/%s_critic_target.pth" % (directory, filename))
        )
        print(f"Loaded weights from: {directory}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        Prepares the environment's raw sensor data and navigation variables into
        a format suitable for learning.

        Args:
            latest_scan (list or np.ndarray): Raw scan data (e.g., LiDAR).
            distance (float): Distance to goal.
            cos (float): Cosine of heading angle to goal.
            sin (float): Sine of heading angle to goal.
            collision (bool): Collision status (True if collided).
            goal (bool): Goal reached status.
            action (list or np.ndarray): Last action taken [lin_vel, ang_vel].

        Returns:
            (tuple):
                - state (list): Normalized and concatenated state vector.
                - terminal (int): Terminal flag (1 if collision or goal, else 0).
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        latest_scan /= 7

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        # state = latest_scan.tolist() + [lin_vel, ang_vel]

        # assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0
        state = {
            "lidar": latest_scan.tolist(),
            "goal": [distance, cos, sin],
            "velocity": [lin_vel, ang_vel]
        }

        return state, terminal
