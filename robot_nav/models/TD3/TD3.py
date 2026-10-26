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
    Actor network for the TD3 algorithm.

    This neural network maps states to actions using a feedforward architecture with
    LeakyReLU activations and a final Tanh output to bound the actions in [-1, 1].

    Architecture:
        Input: state_dim
        Hidden Layer 1: 400 units, LeakyReLU
        Hidden Layer 2: 300 units, LeakyReLU
        Output Layer: action_dim, Tanh

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the action output.
    """

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Perform a forward pass through the actor network.

        Args:
            s (torch.Tensor): Input state tensor.

        Returns:
            (torch.Tensor): Action output tensor after Tanh activation.
        """
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """
    Critic network for the TD3 algorithm.

    This class defines two Q-value estimators (Q1 and Q2) using separate subnetworks.
    Each Q-network takes both state and action as input and outputs a scalar Q-value.

    Architecture for each Q-network:
        Input: state_dim and action_dim
        - State pathway: Linear + LeakyReLU → 400 → 300
        - Action pathway: Linear → 300
        - Combined pathway: LeakyReLU(Linear(state) + Linear(action) + bias) → 1

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the input action.
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, a):
        """
        Perform a forward pass through both Q-networks.

        Args:
            s (torch.Tensor): Input state tensor.
            a (torch.Tensor): Input action tensor.

        Returns:
            (tuple):
                - q1 (torch.Tensor): Output Q-value from the first critic network.
                - q2 (torch.Tensor): Output Q-value from the second critic network.
        """
        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/TD3/checkpoint"),
        model_name="TD3",
        load_directory=Path("robot_nav/models/TD3/checkpoint"),
        use_max_bound=False,
        bound_weight=0.25,
    ):
        """
        Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

        This class implements the TD3 reinforcement learning algorithm for continuous control.
        It uses an Actor-Critic architecture with target networks and delayed policy updates.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            max_action (float): Maximum allowed value for actions.
            device (torch.device): Device to run the model on (CPU or CUDA).
            lr (float, optional): Learning rate for both actor and critic. Default is 1e-4.
            save_every (int, optional): Save model every `save_every` iterations. Default is 0.
            load_model (bool, optional): Whether to load model from checkpoint. Default is False.
            save_directory (Path, optional): Directory to save model checkpoints.
            model_name (str, optional): Name to use when saving/loading models.
            load_directory (Path, optional): Directory to load model checkpoints from.
            use_max_bound (bool, optional): Whether to apply maximum Q-value bounding during training.
            bound_weight (float, optional): Weight for the max-bound loss penalty.
        """
        self.device = device
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
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
        Get an action from the current policy with optional exploration noise.

        Args:
            obs (np.ndarray): The current state observation.
            add_noise (bool): Whether to add exploration noise.

        Returns:
            (np.ndarray): The chosen action clipped to [-max_action, max_action].
        """
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.15, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        """
        Compute the action using the actor network without exploration noise.

        Args:
            state (np.ndarray): The current environment state.

        Returns:
            (np.ndarray): The deterministic action predicted by the actor.
        """
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
        Train the TD3 agent using batches sampled from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample experiences from.
            iterations (int): Number of training iterations to perform.
            batch_size (int): Size of each mini-batch.
            discount (float): Discount factor gamma for future rewards.
            tau (float): Soft update rate for target networks.
            policy_noise (float): Stddev of Gaussian noise added to target actions.
            noise_clip (float): Maximum magnitude of noise added to target actions.
            policy_freq (int): Frequency of policy (actor) updates.
            max_lin_vel (float): Max linear velocity used for upper bound estimation.
            max_ang_vel (float): Max angular velocity used for upper bound estimation.
            goal_reward (float): Reward given for reaching the goal.
            distance_norm (float): Distance normalization factor.
            time_step (float): Time step used in upper bound calculations.
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
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
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
        Save the actor and critic networks (and their targets) to disk.

        Args:
            filename (str): Name to use when saving model files.
            directory (Path): Directory where models should be saved.
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
        Load the actor and critic networks (and their targets) from disk.

        Args:
            filename (str): Name used when saving the models.
            directory (Path): Directory where models are saved.
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
        Prepare the input state vector for training or inference.

        Combines processed laser scan data, goal vector, and past action
        into a normalized state input matching the input dimension.

        Args:
            latest_scan (list or np.ndarray): Laser scan data.
            distance (float): Distance to goal.
            cos (float): Cosine of the heading angle to goal.
            sin (float): Sine of the heading angle to goal.
            collision (bool): Whether a collision occurred.
            goal (bool): Whether the goal has been reached.
            action (list or np.ndarray): Last executed action [linear_vel, angular_vel].

        Returns:
            (tuple):
                - state (list): Prepared and normalized state vector.
                - terminal (int): 1 if episode should terminate (goal or collision), else 0.
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 7)

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel]

        assert len(state) == self.state_dim, f"{len(state), self.state_dim}"
        terminal = 1 if collision or goal else 0

        return state, terminal
