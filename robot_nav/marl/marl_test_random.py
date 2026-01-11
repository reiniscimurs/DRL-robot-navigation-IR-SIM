import statistics
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from robot_nav.marl.marlTD3.marlTD3 import TD3

import torch
import numpy as np
from robot_nav.SIM_ENV.marl_sim import MARL_SIM


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
    test_scenarios = 1000

    # ---- Instantiate simulation environment and model ----
    sim = MARL_SIM(
        world_file="../worlds/multi_robot_world.yaml",
        disable_plotting=True,
        reward_phase=2,
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

    reward_per_ep = []
    goals_per_ep = []
    col_per_ep = []
    lin_actions = []
    ang_actions = []
    entropy_list = []
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

        # Entropy for analysis/logging
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

        a_in = [
            [(a[0] + 1) / 4, a[1]] for a in action
        ]  # clip linear velocity to [0, 0.5] m/s range

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
        running_collisions += sum(collision) / 2
        running_reward += sum(reward)
        running_timesteps += 1
        for j in range(len(a_in)):
            lin_actions.append(a_in[j][0].item())
            ang_actions.append(a_in[j][1].item())
        outside = outside_of_bounds(poses)

        if (
            sum(collision) > 0.5 or steps == max_steps or outside
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

            steps = 0
            episode += 1
            pbar.update(1)
        else:
            steps += 1

    reward_per_ep = np.array(reward_per_ep, dtype=np.float32)
    goals_per_ep = np.array(goals_per_ep, dtype=np.float32)
    col_per_ep = np.array(col_per_ep, dtype=np.float32)
    lin_actions = np.array(lin_actions, dtype=np.float32)
    ang_actions = np.array(ang_actions, dtype=np.float32)
    entropy = np.array(entropy_list, dtype=np.float32)
    avg_ep_reward = statistics.mean(reward_per_ep)
    avg_ep_reward_std = statistics.stdev(reward_per_ep)
    avg_ep_col = statistics.mean(col_per_ep)
    avg_ep_col_std = statistics.stdev(col_per_ep)
    avg_ep_goals = statistics.mean(goals_per_ep)
    avg_ep_goals_std = statistics.stdev(goals_per_ep)
    mean_lin_action = statistics.mean(lin_actions)
    lin_actions_std = statistics.stdev(lin_actions)
    mean_ang_action = statistics.mean(ang_actions)
    ang_actions_std = statistics.stdev(ang_actions)
    mean_entropy = statistics.mean(entropy)
    mean_entropy_std = statistics.stdev(entropy)

    print(f"avg_ep_reward: {avg_ep_reward}")
    print(f"avg_ep_reward_std: {avg_ep_reward_std}")
    print(f"avg_ep_col: {avg_ep_col}")
    print(f"avg_ep_col_std: {avg_ep_col_std}")
    print(f"avg_ep_goals: {avg_ep_goals}")
    print(f"avg_ep_goals_std: {avg_ep_goals_std}")
    print(f"mean_lin_action: {mean_lin_action}")
    print(f"lin_actions_std: {lin_actions_std}")
    print(f"mean_ang_action: {mean_ang_action}")
    print(f"ang_actions_std: {ang_actions_std}")
    print(f"mean_entropy: {mean_entropy}")
    print(f"mean_entropy_std: {mean_entropy_std}")
    print("..............................................")
    model.writer.add_scalar("test/avg_ep_reward", avg_ep_reward, epoch)
    model.writer.add_scalar("test/avg_ep_reward_std", avg_ep_reward_std, epoch)
    model.writer.add_scalar("test/avg_ep_col", avg_ep_col, epoch)
    model.writer.add_scalar("test/avg_ep_col_std", avg_ep_col_std, epoch)
    model.writer.add_scalar("test/avg_ep_goals", avg_ep_goals, epoch)
    model.writer.add_scalar("test/avg_ep_goals_std", avg_ep_goals_std, epoch)
    model.writer.add_scalar("test/mean_lin_action", mean_lin_action, epoch)
    model.writer.add_scalar("test/lin_actions_std", lin_actions_std, epoch)
    model.writer.add_scalar("test/mean_ang_action", mean_ang_action, epoch)
    model.writer.add_scalar("test/ang_actions_std", ang_actions_std, epoch)
    model.writer.add_scalar("test/mean_entropy", mean_entropy, epoch)
    model.writer.add_scalar("test/mean_entropy_std", mean_entropy_std, epoch)
    bins = 100
    model.writer.add_histogram("test/lin_actions", lin_actions, epoch, max_bins=bins)
    model.writer.add_histogram("test/ang_actions", ang_actions, epoch, max_bins=bins)

    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/lin_actions_hist", fig)

    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/ang_actions_hist", fig)


if __name__ == "__main__":
    main()
