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

        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(26, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Forward pass through the Actor network.

        Args:
            s (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
                              The last 5 elements are [distance, cos, sin, lin_vel, ang_vel].

        Returns:
            (torch.Tensor): Action tensor of shape (batch_size, action_dim),
                          with values in range [-1, 1] due to tanh activation.
        """
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        laser = s[:, :-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, a), dim=-1)

        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


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

        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(26, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(26, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, action):
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
        laser = s[:, :-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, a), dim=-1)

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
class devCNNTD3(object):
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
        save_directory=Path("robot_nav/models/CNNTD3/checkpoint"),
        model_name="devCNNTD3",
        load_directory=Path("robot_nav/models/CNNTD3/checkpoint"),
        use_max_bound=False,
        bound_weight=0.25,
    ):
        # Initialize the Actor network
        self.device = device
        self.actor = Actor(action_dim).to(self.device)
        self.actor_target = Actor(action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(action_dim).to(self.device)
        self.critic_target = Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

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
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
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
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

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
        max_lin_vel=0.5,
        max_ang_vel=1,
        goal_reward=100,
        distance_norm=10,
        time_step=0.3,
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
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device).reshape(-1, 1)
            done = torch.Tensor(batch_dones).to(self.device).reshape(-1, 1)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            if self.use_max_bound:
                max_bound = get_max_bound(
                    next_state,
                    discount,
                    max_ang_vel,
                    max_lin_vel,
                    time_step,
                    distance_norm,
                    goal_reward,
                    reward,
                    done,
                    self.device,
                )
                max_excess_Q1 = F.relu(current_Q1 - max_bound)
                max_excess_Q2 = F.relu(current_Q2 - max_bound)
                max_bound_loss = (max_excess_Q1**2).mean() + (max_excess_Q2**2).mean()
                # Add loss for Q values exceeding maximum possible upper bound
                loss += self.bound_weight * max_bound_loss

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
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
        state = latest_scan.tolist() + [lin_vel, ang_vel]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
