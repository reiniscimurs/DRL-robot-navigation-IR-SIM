import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from numpy import inf
import torch.nn.functional as F


class RolloutBuffer:
    """
    Buffer to store rollout data (transitions) for PPO training.

    Attributes:
        actions (list): Actions taken by the agent.
        states (list): States observed by the agent.
        logprobs (list): Log probabilities of the actions.
        rewards (list): Rewards received from the environment.
        state_values (list): Value estimates for the states.
        is_terminals (list): Flags indicating episode termination.
    """

    def __init__(self):
        """
        Initialize empty lists to store buffer elements.
        """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        """
        Clear all stored data from the buffer.
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, reward, terminal, next_state):
        """
        Add a transition to the buffer. (Partial implementation.)

        Args:
            state (list or np.array): The current observed state.
            action (list or np.array): The action taken.
            reward (float): The reward received after taking the action.
            terminal (bool): Whether the episode terminated.
            next_state (list or np.array): The resulting state after taking the action.
        """
        self.states.append(state)
        self.rewards.append(reward)
        self.is_terminals.append(terminal)


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
        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
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
        laser = s[:, :-5]
        goal = s[:, -5:-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, g, a), dim=-1)

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

    def __init__(self):
        super(Critic, self).__init__()
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.goal_embed = nn.Linear(3, 10)
        self.action_embed = nn.Linear(2, 10)

        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, s):
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
        if s.dim() < 2:
            s = s.unsqueeze(0)
        laser = s[:, :-5]
        goal = s[:, -5:-2]
        act = s[:, -2:]
        laser = laser.unsqueeze(1)

        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.goal_embed(goal))

        a = F.leaky_relu(self.action_embed(act))

        s = torch.concat((l, g, a), dim=-1)

        s1 = F.leaky_relu(self.layer_1(s))
        s2 = F.leaky_relu(self.layer_2(s1))
        q1 = self.layer_3(s2)
        return q1


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network model for PPO.

    Attributes:
        actor (nn.Sequential): Policy network (actor) to output action mean.
        critic (nn.Sequential): Value network (critic) to predict state values.
        action_var (Tensor): Diagonal covariance matrix for action distribution.
        device (str): Device used for computation ('cpu' or 'cuda').
        max_action (float): Clipping range for action values.
    """

    def __init__(self, state_dim, action_dim, action_std_init, max_action, device):
        """
        Initialize the Actor and Critic networks.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            action_std_init (float): Initial standard deviation of the action distribution.
            max_action (float): Maximum value allowed for an action (clipping range).
            device (str): Device to run the model on.
        """
        super(ActorCritic, self).__init__()

        self.device = device
        self.max_action = max_action

        self.action_dim = action_dim
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(self.device)
        # actor
        self.actor = Actor(action_dim)
        # critic
        self.critic = Critic()

    def set_action_std(self, new_action_std):
        """
        Set a new standard deviation for the action distribution.

        Args:
            new_action_std (float): New standard deviation.
        """
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(self.device)

    def forward(self):
        """
        Forward method is not implemented, as it's unused directly.

        Raises:
            NotImplementedError: Always raised when called.
        """
        raise NotImplementedError

    def act(self, state, sample):
        """
        Compute an action, its log probability, and the state value.

        Args:
            state (Tensor): Input state tensor.
            sample (bool): Whether to sample from the action distribution or use mean.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Sampled (or mean) action, log probability, and state value.
        """
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        if sample:
            action = torch.clip(
                dist.sample(), min=-self.max_action, max=self.max_action
            )
        else:
            action = dist.mean
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, entropy, and state values for given states and actions.

        Args:
            state (Tensor): Batch of states.
            action (Tensor): Batch of actions.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): Action log probabilities, state values, and distribution entropy.
        """
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class CNNPPO:
    """
    Proximal Policy Optimization (PPO) implementation for continuous control tasks.

    Attributes:
        max_action (float): Maximum action value.
        action_std (float): Standard deviation of the action distribution.
        action_std_decay_rate (float): Rate at which to decay action standard deviation.
        min_action_std (float): Minimum allowed action standard deviation.
        state_dim (int): Dimension of the state space.
        gamma (float): Discount factor for future rewards.
        eps_clip (float): Clipping range for policy updates.
        device (str): Device for model computation ('cpu' or 'cuda').
        save_every (int): Interval (in iterations) for saving model checkpoints.
        model_name (str): Name used when saving/loading model.
        save_directory (Path): Directory to save model checkpoints.
        iter_count (int): Number of training iterations completed.
        buffer (RolloutBuffer): Buffer to store trajectories.
        policy (ActorCritic): Current actor-critic network.
        optimizer (torch.optim.Optimizer): Optimizer for actor and critic.
        policy_old (ActorCritic): Old actor-critic network for computing PPO updates.
        MseLoss (nn.Module): Mean squared error loss function.
        writer (SummaryWriter): TensorBoard summary writer.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        eps_clip=0.2,
        action_std_init=0.6,
        action_std_decay_rate=0.0015,
        min_action_std=0.15,
        device="cpu",
        save_every=10,
        load_model=False,
        save_directory=Path("robot_nav/models/CNNPPO/checkpoint"),
        model_name="CNNPPO",
        load_directory=Path("robot_nav/models/CNNPPO/checkpoint"),
    ):
        self.max_action = max_action
        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.iter_count = 0

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter(comment=model_name)

    def set_action_std(self, new_action_std):
        """
        Set a new standard deviation for the action distribution.

        Args:
            new_action_std (float): New standard deviation value.
        """
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """
        Decay the action standard deviation by a fixed rate, down to a minimum threshold.

        Args:
            action_std_decay_rate (float): Amount to reduce standard deviation by.
            min_action_std (float): Minimum value for standard deviation.
        """
        print(
            "--------------------------------------------------------------------------------------------"
        )
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print(
                "setting actor output action_std to min_action_std : ", self.action_std
            )
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def get_action(self, state, add_noise):
        """
        Sample an action using the current policy (optionally with noise), and store in buffer if noise is added.

        Args:
            state (array_like): Input state for the policy.
            add_noise (bool): Whether to sample from the distribution (True) or use the deterministic mean (False).

        Returns:
            (np.ndarray): Sampled action.
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, add_noise)

        if add_noise:
            # self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size):
        """
        Train the policy and value function using PPO loss based on the stored rollout buffer.

        Args:
            replay_buffer (object): Placeholder for compatibility (not used).
            iterations (int): Number of epochs to optimize the policy per update.
            batch_size (int): Batch size (not used; training uses the whole buffer).
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        assert len(self.buffer.actions) == len(self.buffer.states)

        states = [torch.tensor(st, dtype=torch.float32) for st in self.buffer.states]
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(self.device)
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        av_state_values = 0
        max_state_value = -inf
        av_loss = 0
        # Optimize policy for K epochs
        for _ in range(iterations):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            av_state_values += torch.mean(state_values)
            max_state_value = max(max_state_value, max(state_values))
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            av_loss += loss.mean()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/avg_value", av_state_values / iterations, self.iter_count
        )
        self.writer.add_scalar("train/max_value", max_state_value, self.iter_count)
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

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
        state = latest_scan.tolist() + [distance, cos, sin] + [lin_vel, ang_vel]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal

    def save(self, filename, directory):
        """
        Save the current policy model to the specified directory.

        Args:
            filename (str): Base name of the model file.
            directory (Path): Directory to save the model to.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(
            self.policy_old.state_dict(), "%s/%s_policy.pth" % (directory, filename)
        )

    def load(self, filename, directory):
        """
        Load the policy model from a saved checkpoint.

        Args:
            filename (str): Base name of the model file.
            directory (Path): Directory to load the model from.
        """
        self.policy_old.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        self.policy.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,
            )
        )
        print(f"Loaded weights from: {directory}")
