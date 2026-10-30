import random
from collections import deque
import itertools

import numpy as np


class ReplayBuffer(object):
    """
    Standard experience replay buffer for off-policy reinforcement learning algorithms.

    Stores tuples of (state, action, reward, done, next_state) up to a fixed capacity,
    enabling sampling of uncorrelated mini-batches for training.
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            random_seed (int): Seed for random number generation.
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """
        Add a transition to the buffer.

        Args:
            s (np.ndarray): State.
            a (np.ndarray): Action.
            r (float): Reward.
            t (bool): Done flag (True if episode ended).
            s2 (np.ndarray): Next state.
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            (int): Current buffer size.
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            (Tuple of np.ndarrays): Batches of states, actions, rewards, done flags, and next states.
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        """
        Return the entire buffer contents as separate arrays.

        Returns:
            (Tuple of np.ndarrays): Full arrays of states, actions, rewards, done flags, and next states.
        """
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        """
        Clear all contents of the buffer.
        """
        self.buffer.clear()
        self.count = 0


class RolloutReplayBuffer(object):
    """
    Replay buffer that stores full episode rollouts, allowing access to historical trajectories.

    Useful for algorithms that condition on sequences of past states (e.g., RNN-based policies).
    """

    def __init__(self, buffer_size, random_seed=123, history_len=10):
        """
        Initialize the rollout replay buffer.

        Args:
            buffer_size (int): Maximum number of episodes (rollouts) to store.
            random_seed (int): Seed for random number generation.
            history_len (int): Number of past steps to return for each sampled state.
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)
        self.buffer.append([])
        self.history_len = history_len

    def add(self, s, a, r, t, s2):
        """
        Add a transition to the current episode.

        If the transition ends the episode (t=True), a new episode is started.

        Args:
            s (np.ndarray): State.
            a (np.ndarray): Action.
            r (float): Reward.
            t (bool): Done flag.
            s2 (np.ndarray): Next state.
        """
        experience = (s, a, r, t, s2)
        if t:
            self.count += 1
            self.buffer[-1].append(experience)
            self.buffer.append([])
        else:
            self.buffer[-1].append(experience)

    def size(self):
        """
        Get the number of complete episodes in the buffer.

        Returns:
            (int): Number of episodes.
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of state sequences and corresponding transitions from full episodes.

        Returns past `history_len` steps for each sampled transition, padded with the earliest step if necessary.

        Args:
            batch_size (int): Number of sequences to sample.

        Returns:
            (Tuple of np.ndarrays): Sequences of past states, actions, rewards, done flags, and next states.
        """
        if self.count < batch_size:
            batch = random.sample(
                list(itertools.islice(self.buffer, 0, len(self.buffer) - 1)), self.count
            )
        else:
            batch = random.sample(
                list(itertools.islice(self.buffer, 0, len(self.buffer) - 1)), batch_size
            )

        idx = [random.randint(0, len(b) - 1) for b in batch]

        s_batch = []
        s2_batch = []
        for i in range(len(batch)):
            if idx[i] == len(batch[i]):
                s = batch[i]
                s2 = batch[i]
            else:
                s = batch[i][: idx[i] + 1]
                s2 = batch[i][: idx[i] + 1]
            s = [v[0] for v in s]
            s = s[::-1]

            s2 = [v[4] for v in s2]
            s2 = s2[::-1]

            if len(s) < self.history_len:
                missing = self.history_len - len(s)
                s += [s[-1]] * missing
                s2 += [s2[-1]] * missing
            else:
                s = s[: self.history_len]
                s2 = s2[: self.history_len]
            s = s[::-1]
            s_batch.append(s)
            s2 = s2[::-1]
            s2_batch.append(s2)

        a_batch = np.array([batch[i][idx[i]][1] for i in range(len(batch))])
        r_batch = np.array([batch[i][idx[i]][2] for i in range(len(batch))]).reshape(
            -1, 1
        )
        t_batch = np.array([batch[i][idx[i]][3] for i in range(len(batch))]).reshape(
            -1, 1
        )

        return np.array(s_batch), a_batch, r_batch, t_batch, np.array(s2_batch)

    def clear(self):
        """
        Clear all stored episodes from the buffer.
        """
        self.buffer.clear()
        self.count = 0


class JointReplayBuffer(ReplayBuffer):
    """One transition per scene tick; never sample robots independently.

    Samples have shapes (B,N,D), (B,N,A), (B,N), (B,N), (B,N,D).
    Copies protect stored scenes from later environment/caller mutations.
    """

    def add(self, s, a, r, t, s2):
        s, a, r, t, s2 = [np.array(v, copy=True) for v in (s, a, r, t, s2)]
        if (s.ndim != 2 or s2.shape != s.shape or a.ndim != 2
                or a.shape[0] != len(s) or r.shape != (len(s),)
                or t.shape != (len(s),)):
            raise ValueError("Expected synchronized states/actions and per-agent rewards/dones")
        super().add(s, a, r, t, s2)
