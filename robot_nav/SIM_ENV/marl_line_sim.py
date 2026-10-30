"""Synchronized multi-robot path following with IGA-compatible observations."""
from pathlib import Path

import irsim
import numpy as np

from robot_nav.SIM_ENV.line_sim import SIM
from robot_nav.SIM_ENV.trajectory import SplineTrajectory


class MARL_LINE_SIM:
    def __init__(self, world_file=None, disable_plotting=False):
        if world_file is None:
            world_file = Path(__file__).resolve().parents[1] / "worlds/marl_line_world.yaml"
        self.env = irsim.make(str(world_file), disable_all_plot=disable_plotting,
                              display=not disable_plotting)
        self.num_robots = len(self.env.robot_list)
        config = self.env.config["custom"]
        self.lookahead_distances = np.asarray(config["lookahead_distances"], dtype=float)
        if (self.lookahead_distances.ndim != 1
                or not np.isfinite(self.lookahead_distances).all()
                or np.any(self.lookahead_distances < 0)):
            raise ValueError("lookahead_distances must be finite and nonnegative")
        self.base_trajectories = np.asarray(config["trajectories"], dtype=float)
        self.trajectories = self._paths(self.base_trajectories)

    def _paths(self, trajectories):
        if len(trajectories) != self.num_robots:
            raise ValueError("Each robot requires its own trajectory")
        return [p if isinstance(p, SplineTrajectory) else SplineTrajectory(p)
                for p in trajectories]

    @property
    def state_dim(self):
        return 12 + 2 * len(self.lookahead_distances)

    def reset(self, trajectories=None, randomize=True):
        """Reset all robots together, optionally varying each reference path."""
        if trajectories is None:
            trajectories = self.base_trajectories.copy()
            if randomize:
                trajectories[:, 1:-1, :] += np.random.uniform(
                    -3, 3, trajectories[:, 1:-1, :].shape)
        self.trajectories = self._paths(trajectories)
        for robot, path in zip(self.env.robot_list, self.trajectories):
            start = path.point_at(0)
            heading = float(path.heading_at(0))
            offset = np.random.uniform(-0.5, 0.5) if randomize else 0
            angle = np.random.uniform(-0.3, 0.3) if randomize else 0
            pose = [start[0] - np.sin(heading) * offset,
                    start[1] + np.cos(heading) * offset, heading + angle]
            robot.set_state(np.asarray(pose).reshape(3, 1), init=True)
            robot.set_goal(path.point_at(path.length).reshape(2, 1), init=True)
        self.env.reset()
        return self._observe(np.zeros((self.num_robots, 2)))[0]

    def step(self, actions, combined_weights=None):
        """Step all robots; optionally draw IGA weights [receiver, sender]."""
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_robots, 2) or not np.isfinite(actions).all():
            raise ValueError("actions must have shape (num_robots, 2) and be finite")
        self.env.step(action_id=list(range(self.num_robots)),
                      action=[a.reshape(2, 1) for a in actions])
        self.env.render()
        observation = self._observe(actions)
        if combined_weights is not None:
            self._draw_connections(combined_weights)
        return observation

    def _draw_connections(self, combined_weights):
        """Match MARL's strength-scaled lines, colored by the receiver.

        Opposite directions are offset slightly so both remain visible.
        Weights are gate times soft attention, as used by message passing.
        """
        if self.env.disable_all_plot or not self.env._world.sampling:
            return
        if hasattr(combined_weights, "detach"):
            combined_weights = combined_weights.detach().cpu().numpy()
        weights = np.asarray(combined_weights)
        if weights.shape != (self.num_robots, self.num_robots):
            raise ValueError("combined_weights must have shape (num_robots, num_robots)")
        for i, receiver in enumerate(self.env.robot_list):
            end = receiver.state[:2].reshape(2)
            for j, sender in enumerate(self.env.robot_list):
                strength = float(np.clip(weights[i, j], 0, 1))
                if i == j or strength <= 0:
                    continue
                start = sender.state[:2].reshape(2)
                direction = end - start
                offset = 0.08 * np.array([-direction[1], direction[0]]) / (
                    np.linalg.norm(direction) + 1e-10)
                self.env.draw_trajectory(
                    np.stack([start + offset, end + offset], axis=1),
                    traj_type="-", color=receiver.color, linewidth=4 * strength,
                    alpha=0.2 + 0.8 * strength, refresh=True, zorder=4,
                )

    def _observe(self, actions):
        states, rewards, terminals = [], [], []
        for robot, path, action in zip(self.env.robot_list, self.trajectories, actions):
            x, y, theta = robot.state[:3].reshape(-1)
            position = np.array([x, y])
            projection, progress = path.project(position)
            vector = projection - position
            distance = float(np.linalg.norm(vector))
            heading = np.array([np.cos(theta), np.sin(theta)])
            cos, sin = SIM.cossin(heading, vector)
            hd = (path.heading_at(progress) - theta + np.pi) % (2 * np.pi) - np.pi
            lookahead = path.lookahead_points(progress, self.lookahead_distances)
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            local_points = (lookahead - position) @ rotation
            # Keep IGA's pose, five local features, and final target coordinates.
            # Extra local path features sit before the final two coordinates.
            target = path.point_at(min(progress + 3, path.length))
            state = [x, y, *heading, distance, cos, sin, *action, hd,
                     *local_points.ravel(), *target]
            reached = np.linalg.norm(position - path.point_at(path.length)) < 0.3
            outside = not (self.env._world.x_range[0] <= x <= self.env._world.x_range[1]
                           and self.env._world.y_range[0] <= y <= self.env._world.y_range[1])
            collision = bool(robot.collision)
            reward = SIM.get_reward(distance, hd, *action) - 20 * collision
            states.append(state)
            rewards.append(reward)
            terminals.append(collision or reached or outside)
            if not self.env.disable_all_plot and self.env._world.sampling:
                self.env.draw_trajectory(path.sample_points().T, traj_type="-",
                                         color=robot.color, refresh=True, linewidth=1)
                self.env.draw_points(
                    np.asarray(projection).reshape(2, 1), s=45, c=robot.color,
                    marker="o", refresh=True, zorder=5,
                )
                if lookahead.size:
                    self.env.draw_points(
                        lookahead.T, s=55, c=robot.color, marker="x",
                        refresh=True, zorder=5,
                    )
        return (np.asarray(states, dtype=np.float32), np.asarray(rewards, dtype=np.float32),
                np.asarray(terminals, dtype=bool))
