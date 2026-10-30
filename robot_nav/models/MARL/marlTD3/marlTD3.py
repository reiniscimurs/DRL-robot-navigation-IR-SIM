from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from robot_nav.models.MARL.Attention.g2anet import G2ANet
from robot_nav.models.MARL.Attention.iga import Attention


class Actor(nn.Module):
    """
    Policy network for multi-agent control with an attention encoder.

    The actor encodes inter-agent context (via IGA or G2ANet attention) and maps
    the attended embedding to continuous actions.

    Args:
        action_dim (int): Number of action dimensions per agent.
        embedding_dim (int): Dimensionality of the attention embedding.
        attention (str): Attention backend, one of {"igs", "g2anet"}.

    Attributes:
        attention (nn.Module): Attention encoder producing attended embeddings and
            diagnostics (hard logits, distances, entropy, masks, combined weights).
        policy_head (nn.Sequential): MLP mapping attended embeddings to actions in [-1, 1].
    """

    def __init__(self, action_dim, embedding_dim, attention, local_state_dim=5):
        super().__init__()
        if attention in {"iga", "igs"}:
            self.attention = Attention(embedding_dim, local_state_dim=local_state_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)  # ➊ edge classifier
        else:
            raise ValueError("unknown attention mechanism in Actor")

        # ➋ policy head (everything _after_ attention)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs, detach_attn=False):
        """
        Run the actor to produce actions and attention diagnostics.

        Args:
            obs (Tensor): Observations of shape (B, N, obs_dim) or (N, obs_dim).
            detach_attn (bool, optional): If True, detaches attention features before
                the policy head (useful for target policy smoothing). Defaults to False.

        Returns:
            tuple:
                action (Tensor): Predicted actions, shape (B*N, action_dim).
                hard_logits (Tensor): Hard attention logits, shape (B*N, N-1).
                pair_d (Tensor): Unnormalized pairwise distances, shape (B*N, N-1, 1).
                mean_entropy (Tensor): IGA selector plus soft-attention entropy.
                hard_weights (Tensor): Binary hard attention mask, shape (B, N, N).
                combined_weights (Tensor): Soft weights per (receiver, sender),
                    shape (N, N*(N-1)) for each batch item (batched internally).
        """
        attn_out, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights = (
            self.attention(obs)
        )
        if detach_attn:  # used in the policy phase
            attn_out = attn_out.detach()
        action = self.policy_head(attn_out)
        return action, hard_logits, pair_d, mean_entropy, hard_weights, combined_weights


class Critic(nn.Module):
    """
    Twin Q-value critic with attention-based state encoding.

    Computes two independent Q estimates (Q1, Q2) from attended embeddings and
    concatenated actions, following the TD3 design.

    Args:
        action_dim (int): Number of action dimensions per agent.
        embedding_dim (int): Dimensionality of the attention embedding.
        attention (str): Attention backend, one of {"igs", "g2anet"}.

    Attributes:
        attention (nn.Module): Attention encoder producing attended embeddings and
            diagnostics (hard logits, distances, entropy, masks).
        layer_1..layer_6 (nn.Linear): MLP layers forming the twin Q networks.
    """

    def __init__(self, action_dim, embedding_dim, attention, local_state_dim=5):
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        if attention in {"iga", "igs"}:
            self.attention = Attention(embedding_dim, local_state_dim=local_state_dim)
        elif attention == "g2anet":
            self.attention = G2ANet(embedding_dim)  # ➊ edge classifier
        else:
            raise ValueError("unknown attention mechanism in Critic")

        self.layer_1 = nn.Linear(self.embedding_dim * 2, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")

        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")

        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")

        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(self.embedding_dim * 2, 400)
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")

        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")

        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")

        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, embedding, action):
        """
        Compute twin Q values from attended embeddings and actions.

        Args:
            embedding (Tensor): Agent embeddings of shape (B, N, state_dim).
            action (Tensor): Actions of shape (B*N, action_dim).

        Returns:
            tuple:
                Q1 (Tensor): First Q-value estimate, shape (B*N, 1).
                Q2 (Tensor): Second Q-value estimate, shape (B*N, 1).
                mean_entropy (Tensor): IGA selector plus soft-attention entropy.
                hard_logits (Tensor): Hard attention logits, shape (B*N, N-1).
                unnorm_rel_dist (Tensor): Unnormalized pairwise distances, shape (B*N, N-1, 1).
                hard_weights (Tensor): Binary hard attention mask, shape (B, N, N).
        """

        (
            embedding_with_attention,
            hard_logits,
            unnorm_rel_dist,
            mean_entropy,
            hard_weights,
            _,
        ) = self.attention(embedding)

        # Q1
        s1 = F.leaky_relu(self.layer_1(embedding_with_attention))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(action))  # ✅ No .data
        q1 = self.layer_3(s1)

        # Q2
        s2 = F.leaky_relu(self.layer_4(embedding_with_attention))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(action))  # ✅ No .data
        q2 = self.layer_6(s2)

        return q1, q2, mean_entropy, hard_logits, unnorm_rel_dist, hard_weights


class TD3(object):
    """
    Twin Delayed DDPG (TD3) agent for multi-agent continuous control.

    Wraps actor/critic networks, exploration policy, training loop, logging, and
    checkpointing utilities.

    Args:
        state_dim (int): Per-agent state dimension.
        action_dim (int): Per-agent action dimension.
        max_action (float): Action clip magnitude (actions are clipped to [-max_action, max_action]).
        device (torch.device): Target device for models and tensors.
        num_robots (int): Number of agents.
        lr_actor (float, optional): Actor learning rate. Defaults to 1e-4.
        lr_critic (float, optional): Critic learning rate. Defaults to 3e-4.
        save_every (int, optional): Save frequency in training iterations (0 = disabled). Defaults to 0.
        load_model (bool, optional): If True, loads weights on init. Defaults to False.
        save_directory (Path, optional): Directory for saving checkpoints.
        model_name (str, optional): Base filename for checkpoints. Defaults to "marlTD3".
        load_model_name (str or None, optional): Filename base to load. Defaults to None (uses model_name).
        load_directory (Path, optional): Directory to load checkpoints from.
        attention (str, optional): Attention backend, one of {"igs", "g2anet"}. Defaults to "igs".

    Attributes:
        actor (Actor): Policy network.
        actor_target (Actor): Target policy network.
        critic (Critic): Twin Q-value network.
        critic_target (Critic): Target critic network.
        actor_optimizer (torch.optim.Optimizer): Optimizer over attention + policy head.
        critic_optimizer (torch.optim.Optimizer): Optimizer over critic.
        writer (SummaryWriter): TensorBoard writer.
        iter_count (int): Training iteration counter for logging/saving.
        save_every (int): Save frequency.
        model_name (str): Base filename for checkpoints.
        save_directory (Path): Directory for saving checkpoints.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        num_robots,
        lr_actor=1e-4,
        lr_critic=3e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        model_name="marlTD3",
        load_model_name=None,
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        attention="igs",
    ):
        # Initialize the Actor network
        if attention not in ["iga", "igs", "g2anet"]:
            raise ValueError("unknown attention mechanism specified for TD3 model")
        self.num_robots = num_robots
        self.device = device
        self.actor = Actor(action_dim, embedding_dim=256, attention=attention, local_state_dim=state_dim - 6).to(
            self.device
        )  # Using the updated Actor
        self.actor_target = Actor(
            action_dim, embedding_dim=256, attention=attention, local_state_dim=state_dim - 6
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        if isinstance(self.actor.attention, Attention):
            self.actor.attention.forced_connection_probability = 0.5

        self.attn_params = list(self.actor.attention.parameters())
        self.policy_params = list(self.actor.policy_head.parameters())

        self.actor_optimizer = torch.optim.Adam(
            self.policy_params + self.attn_params, lr=lr_actor
        )  # TD3 policy

        self.critic = Critic(action_dim, embedding_dim=256, attention=attention, local_state_dim=state_dim - 6).to(
            self.device
        )  # Using the updated Critic
        self.critic_target = Critic(
            action_dim, embedding_dim=256, attention=attention, local_state_dim=state_dim - 6
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(), lr=lr_critic
        )
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.gradient_steps = 0
        if load_model_name is None:
            load_model_name = model_name
        if load_model:
            self.load(filename=load_model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def get_action(self, obs, add_noise):
        """
        Compute an action for the given observation, with optional exploration noise.

        Args:
            obs (np.ndarray): Observation array of shape (N, state_dim) or (B, N, state_dim).
            add_noise (bool): If True, adds Gaussian exploration noise and clips to bounds.

        Returns:
            tuple:
                action (np.ndarray): Actions reshaped to (N, action_dim).
                connection_logits (Tensor): Hard attention logits from the actor.
                combined_weights (Tensor): Soft attention weights per (receiver, sender).
        """
        action, connection, combined_weights = self.act(obs)
        if add_noise:
            noise = np.random.normal(0, 0.5, size=action.shape)
            noise = [n / 4 if i % 2 else n for i, n in enumerate(noise)]
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action.reshape(-1, 2), connection, combined_weights

    @torch.no_grad()
    def act(self, state):
        """
        Compute the deterministic action from the current policy.

        Args:
            state (np.ndarray): Observation array of shape (N, state_dim).

        Returns:
            tuple:
                action (np.ndarray): Flattened action vector of shape (N*action_dim,).
                connection_logits (Tensor): Hard attention logits from the actor.
                combined_weights (Tensor): Soft attention weights per (receiver, sender).
        """
        # Function to get the action from the actor
        state = torch.Tensor(state).to(self.device)
        # res = self.attention(state)
        action, connection, _, _, _, combined_weights = self.actor(state)
        return action.cpu().numpy().flatten(), connection, combined_weights

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
        communication_weight=0.01,
        entropy_weight=0.01,
        communication_warmup=10000,
        entropy_decay_steps=10000,
        final_entropy_weight=0.0,
        max_grad_norm=7.0,
        forced_connection_probability=0.5,
        forced_connection_decay_steps=10000,
    ):
        """
        Run a TD3 training loop over sampled mini-batches.

        Args:
            replay_buffer: Buffer supporting ``sample_batch(batch_size)`` -> tuple of arrays.
            iterations (int): Number of gradient steps.
            batch_size (int): Mini-batch size.
            discount (float, optional): Discount factor γ. Defaults to 0.99.
            tau (float, optional): Target network update rate. Defaults to 0.005.
            policy_noise (float, optional): Std of target policy noise. Defaults to 0.2.
            noise_clip (float, optional): Clipping range for target noise. Defaults to 0.5.
            policy_freq (int, optional): Actor update period (in critic steps). Defaults to 2.
            communication_weight: Actor cost per mean peer connection probability.
                The critic has no communication cost. Defaults to 0.01.
            entropy_weight: Initial bonus for selector plus soft attention entropy.
            communication_warmup: Gradient steps to ramp the cost from zero to full.
                Zero enables full cost immediately.
            entropy_decay_steps: Gradient steps to decay entropy to final_entropy_weight.
                Zero disables decay. Schedules persist across train calls.
            final_entropy_weight: Entropy bonus after decay. Defaults to zero.
            max_grad_norm (float, optional): Gradient clipping norm. Defaults to 7.0.
            forced_connection_probability: Initial actor peer-link exploration probability.
            forced_connection_decay_steps: Steps to decay forced links to zero.
                Zero disables forced links. Critic and target gates are unaffected.

        Returns:
            None
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_critic_loss = 0
        av_critic_entropy = []
        av_actor_entropy = []
        av_actor_loss = 0
        av_communication = []

        if iterations < 1 or replay_buffer.size() < 1:
            return

        if not 0 <= forced_connection_probability <= 1:
            raise ValueError("forced_connection_probability must be between zero and one")
        if forced_connection_decay_steps < 0:
            raise ValueError("forced_connection_decay_steps must be nonnegative")

        for it in range(iterations):
            progress = self.gradient_steps
            if isinstance(self.actor.attention, Attention):
                fraction_remaining = (
                    max(1 - progress / forced_connection_decay_steps, 0.0)
                    if forced_connection_decay_steps > 0 else 0.0
                )
                self.actor.attention.forced_connection_probability = (
                    forced_connection_probability * fraction_remaining
                )
            cost_weight = communication_weight * (
                min(progress / communication_warmup, 1.0) if communication_warmup > 0 else 1.0
            )
            fraction = min(progress / entropy_decay_steps, 1.0) if entropy_decay_steps > 0 else 0.0
            bonus_weight = entropy_weight + fraction * (final_entropy_weight - entropy_weight)
            # sample a batch
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            sampled_size = len(batch_states)
            state = (
                torch.Tensor(batch_states)
                .to(self.device)
                .view(sampled_size, self.num_robots, self.state_dim)
            )
            next_state = (
                torch.Tensor(batch_next_states)
                .to(self.device)
                .view(sampled_size, self.num_robots, self.state_dim)
            )
            action = (
                torch.Tensor(batch_actions)
                .to(self.device)
                .view(sampled_size * self.num_robots, self.action_dim)
            )
            reward = (
                torch.Tensor(batch_rewards)
                .to(self.device)
                .view(sampled_size * self.num_robots, 1)
            )
            done = (
                torch.Tensor(batch_dones)
                .to(self.device)
                .view(sampled_size * self.num_robots, 1)
            )

            with torch.no_grad():
                next_action, _, _, _, _, _ = self.actor_target(
                    next_state, detach_attn=True
                )

            # --- Target smoothing ---
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            ).reshape(-1, 2)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # --- Target Q values ---
            target_Q1, target_Q2, _, _, _, _ = self.critic_target(
                next_state, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += target_Q.mean()
            max_Q = max(max_Q, target_Q.max().item())
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # --- Critic update ---
            (
                current_Q1,
                current_Q2,
                mean_entropy,
                hard_logits,
                unnorm_rel_dist,
                hard_weights,
            ) = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            total_loss = critic_loss - bonus_weight * mean_entropy
            av_critic_entropy.append(mean_entropy.item())

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            av_loss += total_loss.item()
            av_critic_loss += critic_loss.item()

            # --- Actor update ---
            if it % policy_freq == 0:

                action, hard_logits, unnorm_rel_dist, mean_entropy, hard_weights, _ = (
                    self.actor(state, detach_attn=False)
                )
                # IGA returns the two-class log-odds; sigmoid gives the
                # underlying probability, independent of Gumbel noise.
                probabilities = hard_logits.sigmoid().view(sampled_size, self.num_robots, -1)
                if probabilities.shape[-1] == self.num_robots:
                    peer_mask = ~torch.eye(self.num_robots, dtype=torch.bool, device=self.device)
                    peers = probabilities[:, peer_mask]
                else:  # G2ANet already omits self-connections.
                    peers = probabilities.reshape(-1)
                communication = peers.mean() if peers.numel() else hard_logits.sum() * 0.0
                av_communication.append(communication.item())

                actor_Q, _, _, _, _, _ = self.critic(state, action)
                actor_loss = -actor_Q.mean()
                total_loss = (
                    actor_loss - bonus_weight * mean_entropy + cost_weight * communication
                )
                av_actor_entropy.append(mean_entropy.item())

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.actor_optimizer.step()

                av_actor_loss += total_loss.item()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
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

            self.gradient_steps += 1

        if isinstance(self.actor.attention, Attention):
            self.actor.attention.forced_connection_probability = (
                forced_connection_probability
                * max(1 - self.gradient_steps / forced_connection_decay_steps, 0.0)
                if forced_connection_decay_steps > 0 else 0.0
            )
            self.writer.add_scalar(
                "train/forced_connection_probability",
                self.actor.attention.forced_connection_probability,
                self.iter_count + 1,
            )

        self.iter_count += 1
        self.writer.add_scalar(
            "train/loss_total", av_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/critic_loss", av_critic_loss / iterations, self.iter_count
        )
        self.writer.add_scalar(
            "train/av_critic_entropy",
            sum(av_critic_entropy) / len(av_critic_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_entropy",
            sum(av_actor_entropy) / len(av_actor_entropy),
            self.iter_count,
        )
        self.writer.add_scalar("train/actor_connection_probability", sum(av_communication) / len(av_communication), self.iter_count)
        self.writer.add_scalar("train/communication_weight", cost_weight, self.iter_count)
        self.writer.add_scalar("train/entropy_weight", bonus_weight, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        self.writer.add_scalar(
            "train/actor_loss",
            av_actor_loss / len(av_actor_entropy),
            self.iter_count,
        )

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

    def prepare_state(
        self, poses, distance, cos, sin, collision, action, goal_positions
    ):
        """
        Convert raw environment outputs into per-agent state vectors.

        Args:
            poses (list): Per-agent poses [[x, y, theta], ...].
            distance (list): Per-agent distances to goal.
            cos (list): Per-agent cos(heading error to goal).
            sin (list): Per-agent sin(heading error to goal).
            collision (list): Per-agent collision flags (bool or {0,1}).
            action (list): Per-agent last actions [[lin_vel, ang_vel], ...].
            goal_positions (list): Per-agent goals [[gx, gy], ...].

        Returns:
            tuple:
                states (list): Per-agent state vectors (length == state_dim).
                terminal (list): Terminal flags (collision or goal), same length as states.
        """
        states = []
        terminal = []

        for i in range(self.num_robots):
            pose = poses[i]  # [x, y, theta]
            goal_pos = goal_positions[i]  # [goal_x, goal_y]
            act = action[i]  # [lin_vel, ang_vel]

            px, py, theta = pose
            gx, gy = goal_pos

            # Heading as cos/sin
            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)

            # Last velocity
            lin_vel = act[0] * 2  # Assuming original range [0, 0.5]
            ang_vel = (act[1] + 1) / 2  # Assuming original range [-1, 1]

            # Final state vector
            state = [
                px,
                py,
                heading_cos,
                heading_sin,
                distance[i] / 17,
                cos[i],
                sin[i],
                lin_vel,
                ang_vel,
                gx,
                gy,
            ]

            assert (
                len(state) == self.state_dim
            ), f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            states.append(state)
            terminal.append(collision[i])

        return states, terminal
