import statistics
from pathlib import Path

import irsim
from tqdm import tqdm

from robot_nav.SIM_ENV.sim_env import SIM_ENV
from robot_nav.marl.marlTD3.marlTD3 import TD3

import torch
import numpy as np


class SINGLE_SIM(SIM_ENV):
    """
    Simulation environment for multi-agent robot navigation using IRSim.

    This class extends the SIM_ENV and provides a wrapper for multi-robot
    simulation and interaction, supporting reward computation and custom reset logic.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for the robots.
        num_robots (int): Number of robots in the environment.
        x_range (tuple): World x-range.
        y_range (tuple): World y-range.
    """

    def __init__(
        self,
        world_file="worlds/circle_world.yaml",
        disable_plotting=False,
        reward_phase=2,
    ):
        """
        Initialize the MARL_SIM environment.

        Args:
            world_file (str, optional): Path to the world configuration YAML file.
            disable_plotting (bool, optional): If True, disables IRSim rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        self.num_robots = len(self.env.robot_list)
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.reward_phase = reward_phase

    def step(self, action, connection, combined_weights=None):
        """
        Perform a simulation step for all robots using the provided actions and connections.

        Args:
            action (list): List of actions for each robot [[lin_vel, ang_vel], ...].
            connection (Tensor): Tensor of shape (num_robots, num_robots-1) containing logits indicating connections between robots.
            combined_weights (Tensor or None, optional): Optional weights for each connection, shape (num_robots, num_robots-1).

        Returns:
            tuple: (
                poses (list): List of [x, y, theta] for each robot,
                distances (list): Distance to goal for each robot,
                coss (list): Cosine of angle to goal for each robot,
                sins (list): Sine of angle to goal for each robot,
                collisions (list): Collision status for each robot,
                goals (list): Goal reached status for each robot,
                action (list): Actions applied,
                rewards (list): Rewards computed,
                positions (list): Current [x, y] for each robot,
                goal_positions (list): Goal [x, y] for each robot,
            )
        """
        act = [
            [0, 0] if self.env.robot_list[i].arrive else action[i]
            for i in range(len(action))
        ]
        self.env.step(action_id=[i for i in range(self.num_robots)], action=act)
        self.env.render()

        poses = []
        distances = []
        coss = []
        sins = []
        collisions = []
        goals = []
        rewards = []
        positions = []
        goal_positions = []
        robot_states = [
            [self.env.robot_list[i].state[0], self.env.robot_list[i].state[1]]
            for i in range(self.num_robots)
        ]
        for i in range(self.num_robots):

            robot_state = self.env.robot_list[i].state
            closest_robots = [
                np.linalg.norm(
                    [
                        robot_states[j][0] - robot_state[0],
                        robot_states[j][1] - robot_state[1],
                    ]
                )
                for j in range(self.num_robots)
                if j != i
            ]
            robot_goal = self.env.robot_list[i].goal
            goal_vector = [
                robot_goal[0].item() - robot_state[0].item(),
                robot_goal[1].item() - robot_state[1].item(),
            ]
            distance = np.linalg.norm(goal_vector)
            goal = self.env.robot_list[i].arrive
            pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
            cos, sin = self.cossin(pose_vector, goal_vector)
            collision = self.env.robot_list[i].collision
            action_i = action[i]
            reward = self.get_reward(
                goal, collision, action_i, closest_robots, distance, self.reward_phase
            )

            position = [robot_state[0].item(), robot_state[1].item()]
            goal_position = [robot_goal[0].item(), robot_goal[1].item()]

            distances.append(distance)
            coss.append(cos)
            sins.append(sin)
            collisions.append(collision)
            goals.append(goal)
            rewards.append(reward)
            positions.append(position)
            poses.append(
                [robot_state[0].item(), robot_state[1].item(), robot_state[2].item()]
            )
            goal_positions.append(goal_position)

            i_probs = torch.sigmoid(
                connection[i]
            )  # connection[i] is logits for "connect" per pair

            i_connections = i_probs.tolist()
            i_connections.insert(i, 0)
            if combined_weights is not None:
                i_weights = combined_weights[i].tolist()
                i_weights.insert(i, 0)

            for j in range(self.num_robots):
                if i_connections[j] > 0.5:
                    if combined_weights is not None:
                        weight = i_weights[j]
                    else:
                        weight = 1
                    other_robot_state = self.env.robot_list[j].state
                    other_pos = [
                        other_robot_state[0].item(),
                        other_robot_state[1].item(),
                    ]
                    rx = [position[0], other_pos[0]]
                    ry = [position[1], other_pos[1]]
                    self.env.draw_trajectory(
                        np.array([rx, ry]), refresh=True, linewidth=weight
                    )

        return (
            poses,
            distances,
            coss,
            sins,
            collisions,
            goals,
            action,
            rewards,
            positions,
            goal_positions,
        )

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment and optionally set robot and obstacle positions.

        Args:
            robot_state (list or None, optional): Initial state for robots as [x, y, theta, speed].
            robot_goal (list or None, optional): Goal position(s) for the robots.
            random_obstacles (bool, optional): If True, randomly position obstacles.
            random_obstacle_ids (list or None, optional): IDs of obstacles to randomize.

        Returns:
            tuple: (
                poses (list): List of [x, y, theta] for each robot,
                distances (list): Distance to goal for each robot,
                coss (list): Cosine of angle to goal for each robot,
                sins (list): Sine of angle to goal for each robot,
                collisions (list): All False after reset,
                goals (list): All False after reset,
                action (list): Initial action ([[0.0, 0.0], ...]),
                rewards (list): Rewards for initial state,
                positions (list): Initial [x, y] for each robot,
                goal_positions (list): Initial goal [x, y] for each robot,
            )
        """
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        con = torch.tensor(
            [[0.0 for _ in range(self.num_robots - 1)] for _ in range(self.num_robots)]
        )
        poses, distance, cos, sin, _, _, action, reward, positions, goal_positions = (
            self.step(action, con)
        )
        return (
            poses,
            distance,
            cos,
            sin,
            [False] * self.num_robots,
            [False] * self.num_robots,
            action,
            reward,
            positions,
            goal_positions,
        )

    @staticmethod
    def get_reward(goal, collision, action, closest_robots, distance, phase=2):
        """
        Calculate the reward for a robot given the current state and action.

        Args:
            goal (bool): Whether the robot reached its goal.
            collision (bool): Whether a collision occurred.
            action (list): [linear_velocity, angular_velocity] applied.
            closest_robots (list): Distances to the closest other robots.
            distance (float): Distance to the goal.
            phase (int, optional): Reward phase/function selector (default: 1).

        Returns:
            float: Computed reward.
        """

        match phase:
            case 1:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 1.5 / distance
                    cl_pen = 0
                    for rob in closest_robots:
                        add = 1.5 - rob if rob < 1.5 else 0
                        cl_pen += add

                    return action[0] - 0.5 * abs(action[1]) - cl_pen + r_dist

            case 2:
                if goal:
                    return 70.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (3 - rob) ** 2 if rob < 3 else 0
                        cl_pen += add

                    return -0.5 * abs(action[1]) - cl_pen

            case _:
                raise ValueError("Unknown reward phase")


def outside_of_bounds(poses):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.

    Returns:
        bool: True if any robot is outside the 21x21 area centered at (6, 6), else False.
    """
    outside = False
    for pose in poses:
        norm_x = pose[0] - 6
        norm_y = pose[1] - 6
        if abs(norm_x) > 10.5 or abs(norm_y) > 10.5:
            outside = True
            break
    return outside


def main(args=None):
    """Main training function"""

    # ---- Hyperparameters and setup ----
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 11  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 1  # starting epoch number
    episode = 0
    max_steps = 600  # maximum number of steps in single episode
    steps = 0  # starting step number
    save_every = 5  # save the model every n training cycles
    test_scenarios = 100

    # ---- Instantiate simulation environment and model ----
    sim = SINGLE_SIM(
        world_file="../worlds/circle_world.yaml", disable_plotting=True, reward_phase=2
    )  # instantiate environment

    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        device=device,
        save_every=save_every,
        load_model=True,
        model_name="TDR-MARL-test",
        load_model_name="TDR-MARL-train",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        attention="igs",
    )  # instantiate a model

    connections = torch.tensor(
        [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(sim.num_robots)]
    )

    # ---- Take initial step in environment ----
    poses, distance, cos, sin, collision, goal, a, reward, positions, goal_positions = (
        sim.step([[0, 0] for _ in range(sim.num_robots)], connections)
    )  # get the initial step state
    running_goals = 0
    running_reward = 0
    running_collisions = 0
    running_timesteps = 0
    ran_out_of_time = 0

    reward_per_ep = []
    goals_per_ep = []
    col_per_ep = []
    lin_actions = []
    ang_actions = []
    entropy_list = []
    timesteps_per_ep = []
    epsilon = 1e-6
    pbar = tqdm(total=test_scenarios)
    while episode < test_scenarios:
        state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        action, connection, combined_weights = model.get_action(
            np.array(state), False
        )  # get an action from the model

        combined_weights_norm = combined_weights / (
            combined_weights.sum(dim=-1, keepdim=True) + epsilon
        )

        entropy = (
            (
                -(combined_weights_norm * (combined_weights_norm + epsilon).log())
                .sum(dim=-1)
                .mean()
            )
            .data.cpu()
            .numpy()
        )
        entropy_list.append(entropy)

        a_in = [[(a[0] + 1) / 4, a[1]] for a in action]

        (
            poses,
            distance,
            cos,
            sin,
            collision,
            goal,
            a,
            reward,
            positions,
            goal_positions,
        ) = sim.step(
            a_in, connection, combined_weights
        )  # get data from the environment
        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_reward += sum(reward)
        running_timesteps += 1
        for j in range(len(a_in)):
            lin_actions.append(a_in[j][0].item())
            ang_actions.append(a_in[j][1].item())
        outside = outside_of_bounds(poses)

        if (
            sum(collision) > 0.5 or steps == max_steps or int(sum(goal)) == len(goal)
        ):  # reset environment of terminal state reached, or max_steps were taken
            (
                poses,
                distance,
                cos,
                sin,
                collision,
                goal,
                a,
                reward,
                positions,
                goal_positions,
            ) = sim.reset()
            reward_per_ep.append(running_reward)
            running_reward = 0
            goals_per_ep.append(running_goals)
            running_goals = 0
            col_per_ep.append(running_collisions)
            running_collisions = 0
            timesteps_per_ep.append(running_timesteps)
            running_timesteps = 0
            if steps == max_steps:
                ran_out_of_time += 1

            steps = 0
            episode += 1
            pbar.update(1)
        else:
            steps += 1

    cols = sum(col_per_ep) / 2
    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep)
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep)
    t_per_ep = np.array(timesteps_per_ep, dtype=np.float32)
    avg_ep_t = statistics.mean(t_per_ep)
    avg_ep_t_std = statistics.stdev(t_per_ep)

    print(f"avg_ep_col: {avg_ep_col}")
    print(f"avg_ep_col_std: {avg_ep_col_std}")
    print(f"success rate: {test_scenarios - cols}")
    print(f"avg_ep_goals: {avg_ep_goals}")
    print(f"avg_ep_goals_std: {avg_ep_goals_std}")
    print(f"avg_ep_t: {avg_ep_t}")
    print(f"avg_ep_t_std: {avg_ep_t_std}")
    print(f"ran out of time: {ran_out_of_time}")
    print("..............................................")


if __name__ == "__main__":
    main()
