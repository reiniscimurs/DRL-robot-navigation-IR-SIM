import irsim
import numpy as np
import random
import math

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal

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
        point_vector = [
            projection_point[0].item() - robot_state[0].item(),
            projection_point[1].item() - robot_state[1].item(),
        ]
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, point_vector)
        distance = np.linalg.norm(point_vector)

        # hd = self.heading_difference(robot_state[2].item(), line_start, line_end)
        hd = robot_state[2].item()

        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(distance, hd, lin_velocity, ang_velocity)
        state = [distance, cos, sin, hd, lin_velocity, ang_velocity]

        return state, action, reward

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
            robot_state = [[5], [15 + random.uniform(-8, 8)], [random.uniform(-1.5, 1.5)]]

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
