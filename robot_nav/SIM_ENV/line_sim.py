import irsim
import numpy as np
import random
import math

from robot_nav.SIM_ENV.sim_env import SIM_ENV
from robot_nav.SIM_ENV.trajectory import SplineTrajectory


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False,
                 trajectory=None, lookahead_distances=None):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
            trajectory: Ordered (x, y) waypoints or a SplineTrajectory. None
                reads custom.trajectory from the world YAML when available,
                otherwise preserves the original infinite straight line.
            lookahead_distances: Arc offsets in world units. For a trajectory,
                append flattened robot-frame (forward, left) points to state.
        """
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=not disable_plotting
        )
        config = self.env.config.get("custom", {}) or {}
        if trajectory is None:
            trajectory = config.get("trajectory")
        if lookahead_distances is None:
            lookahead_distances = config.get("lookahead_distances", ())
        self.trajectory = (trajectory if isinstance(trajectory, SplineTrajectory)
                           else SplineTrajectory(trajectory) if trajectory is not None
                           else None)
        self.lookahead_distances = np.asarray(lookahead_distances, dtype=float)
        if (self.lookahead_distances.ndim != 1
                or not np.all(np.isfinite(self.lookahead_distances))
                or np.any(self.lookahead_distances < 0)):
            raise ValueError("lookahead_distances must be a finite nonnegative sequence")
        if self.trajectory is None and self.lookahead_distances.size:
            raise ValueError("lookahead_distances requires a trajectory")
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self._trajectory_plot_points = (self.trajectory.sample_points().T
                                        if self.trajectory is not None else None)

    @property
    def state_dim(self):
        """Observation size for constructing a matching policy network."""
        return 6 + 2 * len(self.lookahead_distances)

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        self.env.render()

        robot_state = self.env.get_robot_state()
        line_start = [ 5, 15 ]
        line_end = [ 100, 15 ] 
        projection_point = self.project_point_to_line([robot_state[0].item(), robot_state[1].item()], line_start, line_end)
        if self.trajectory is not None:
            projection_point, self.path_distance = self.trajectory.project(
                robot_state[:2].reshape(2))
        point_vector = [
            projection_point[0].item() - robot_state[0].item(),
            projection_point[1].item() - robot_state[1].item(),
        ]
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, point_vector)
        distance = np.linalg.norm(point_vector)

        # hd = self.heading_difference(robot_state[2].item(), line_start, line_end)
        hd = robot_state[2].item()
        if self.trajectory is not None:
            hd = (self.trajectory.heading_at(self.path_distance) - hd + np.pi) % (2 * np.pi) - np.pi
            self.lookahead_points = self.trajectory.lookahead_points(
                self.path_distance, self.lookahead_distances)
            theta = robot_state[2].item()
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            self.lookahead_points_robot = (
                self.lookahead_points - robot_state[:2].reshape(2)) @ rotation
            self._draw_trajectory(projection_point)

        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(distance, hd, lin_velocity, ang_velocity)
        state = [distance, cos, sin, hd, lin_velocity, ang_velocity]
        if self.trajectory is not None:
            state.extend(self.lookahead_points_robot.ravel().tolist())

        return state, action, reward

    def _draw_trajectory(self, projection_point):
        """Draw overlays after render clears the previous frame's artists."""
        if self.env.disable_all_plot:
            return
        # IRSim only clears dynamic artists on world sampling frames. Match
        # that cadence, including when sample_time differs from step_time.
        if not self.env._world.sampling:
            return
        self.env.draw_trajectory(
            self._trajectory_plot_points, traj_type="g-", refresh=True,
            linewidth=2, label="Reference spline", zorder=3,
        )
        self.env.draw_points(
            np.asarray(projection_point).reshape(2, 1), s=45, c="tab:blue",
            refresh=True, label="Projection", zorder=5,
        )
        if self.lookahead_points.size:
            self.env.draw_points(
                self.lookahead_points.T, s=55, c="tab:orange", marker="x",
                refresh=True, label="Lookahead", zorder=5,
            )

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally setting robot and obstacle states.

        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles.
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize.

        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        if robot_state is None:
            if self.trajectory is None:
                robot_state = [[5], [15 + random.uniform(-8, 8)], [random.uniform(-1.5, 1.5)]]
            else:
                start = self.trajectory.point_at(0)
                heading = float(self.trajectory.heading_at(0))
                offset = random.uniform(-8, 8)
                robot_state = [[start[0] - np.sin(heading) * offset],
                               [start[1] + np.cos(heading) * offset],
                               [heading + random.uniform(-1.5, 1.5)]]

        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        # if robot_goal is None:
        #     self.env.robot.set_random_goal(
        #         obstacle_list=self.env.obstacle_list,
        #         init=True,
        #         range_limits=[[1, 1, -3.141592653589793], [9, 9, 3.141592653589793]],
        #     )
        # else:
        #     self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        # self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        state, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return state, action, reward

    @staticmethod
    def heading_difference(heading, line_start, line_end):
        x1, y1 = line_start
        x2, y2 = line_end

        # Heading/direction of the line
        line_heading = math.atan2(y2 - y1, x2 - x1)

        # Difference: line heading relative to current heading
        diff = line_heading - heading

        # Normalize to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        return diff

    @staticmethod
    def project_point_to_line(point, line_start, line_end):
        p = np.array(point, dtype=float)
        a = np.array(line_start, dtype=float)
        b = np.array(line_end, dtype=float)

        direction = b - a
        t = np.dot(p - a, direction) / np.dot(direction, direction)

        projection = a + t * direction
        return projection

    @staticmethod
    def get_reward(distance, hd, lin_velocity, ang_velocity):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.

        Returns:
            (float): Computed reward for the current state.
        """
        on_line  = 1 if distance < 0.2 else 0
        return 2*lin_velocity - 0.5*distance - abs(hd)  - abs(ang_velocity) + 2
