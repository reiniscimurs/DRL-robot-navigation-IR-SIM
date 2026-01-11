import numpy as np

from robot_nav.SIM_ENV.sim import SIM


class DEV_SIM(SIM):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def step(
        self,
        lin_velocity=0.0,
        ang_velocity=0.1,
        override_lin=-0.0,
        override_ang=-0.1,
        switch=0,
    ):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        action = [(lin_velocity + 1) / 4, ang_velocity]
        override_action = [override_lin / 2, override_ang * 2]
        if switch:
            a_in = action[:]
        else:
            a_in = [
                (override_lin / 2) + (lin_velocity + 1) / 4,
                override_ang * 2 + ang_velocity,
            ]
            action = [override_lin, override_ang]
        self.env.step(action_id=0, action=np.array([[a_in[0]], [a_in[1]]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        reward = self.get_reward(
            goal, collision, action, override_action, latest_scan, switch
        )

        return latest_scan, distance, cos, sin, collision, goal, a_in, reward

    @staticmethod
    def get_reward(goal, collision, action, override_action, laser_scan, switch):
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
        if switch:
            if goal:
                return 100.0
            elif collision:
                return -100.0
            else:
                deviation = np.linalg.norm(
                    [action[0] - override_action[0], action[1] - override_action[1]]
                )
                return action[0] - abs(action[1]) / 2 - deviation
        else:
            if collision:
                return -100.0
            else:
                return -(override_action[0] ** 2) - (override_action[1] ** 2)
