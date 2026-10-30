"""Train the existing IGA TD3 on joint multi-robot path-following scenes.

Run: python -m robot_nav.marl_line_train --headless
"""
import argparse

import numpy as np
import torch

from robot_nav.models.MARL.marlTD3.marlTD3 import TD3
from robot_nav.replay_buffer import JointReplayBuffer
from robot_nav.SIM_ENV.marl_line_sim import MARL_LINE_SIM


def physical_actions(actions):
    velocities = np.array(actions, copy=True)
    velocities[:, 0] = (velocities[:, 0] + 1) / 4
    return velocities


def evaluate(model, epoch, sim, eval_episodes=10, max_steps=400):
    total = 0.0
    was_training = model.actor.training
    model.actor.eval()
    try:
        for _ in range(eval_episodes):
            state = sim.reset(randomize=False)
            for _ in range(max_steps):
                actions, _, weights = model.get_action(state, False)
                state, rewards, terminals = sim.step(physical_actions(actions), weights)
                total += rewards.mean()
                if terminals.any():
                    break
    finally:
        model.actor.train(was_training)
    average = total / eval_episodes
    model.writer.add_scalar("eval/avg_reward", average, epoch)
    print(f"Epoch {epoch}: average reward per robot {average:.3f}")
    return average


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--episodes-per-epoch", type=int, default=70)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--training-iterations", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=10)
    config = parser.parse_args(args)
    if any(value < 1 for name, value in vars(config).items() if name != "headless"):
        parser.error("Training counts must be positive")
    sim = MARL_LINE_SIM(disable_plotting=config.headless)
    model = TD3(state_dim=sim.state_dim, action_dim=2, max_action=1,
                num_robots=sim.num_robots,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                attention="iga", save_every=5, model_name="IGA-line")
    replay = JointReplayBuffer(100_000)
    episode = 0
    try:
        for epoch in range(1, config.epochs + 1):
            for _ in range(config.episodes_per_epoch):
                state = sim.reset()
                for _ in range(config.max_steps):
                    actions, _, weights = model.get_action(state, True)
                    next_state, rewards, terminals = sim.step(physical_actions(actions), weights)
                    # A failure/completion resets the entire scene. Mark all agents
                    # terminal on that transition; time limits still bootstrap.
                    scene_done = terminals.any()
                    replay.add(state, actions, rewards,
                               np.full(sim.num_robots, scene_done), next_state)
                    state = next_state
                    if scene_done:
                        break
                episode += 1
                if episode % 2 == 0 and replay.size() >= config.batch_size:
                    model.train(replay, config.training_iterations, config.batch_size)
            evaluate(model, epoch, sim, config.eval_episodes, config.max_steps)
        model.save(model.model_name, model.save_directory)
    finally:
        model.writer.close()
        sim.env.end()


if __name__ == "__main__":
    main()
