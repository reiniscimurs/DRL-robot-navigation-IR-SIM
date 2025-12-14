from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.dev_rl.CNNTD3.devCNNTD3 import devCNNTD3

import torch
import numpy as np
from robot_nav.SIM_ENV.dev_sim import DEV_SIM
from robot_nav.utils import get_buffer
import math

def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 182  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 10  # how many episodes to use to run evaluation
    max_epochs = 60  # max number of epochs
    epoch = 1  # starting epoch number
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 2  # train and update network parameters every n episodes
    training_iterations = 80  # how many batches to use for single training cycle
    batch_size = 64  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        10  # number of training iterations to run during pre-training
    )
    save_every = 5  # save the model every n training cycles

    deviation_model = devCNNTD3(
        state_dim=182,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="devCNNTD3",
    )  # instantiate a model

    train_model = CNNTD3(
        state_dim=185,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=True,
        model_name="fullCNNTD3",
    )  # instantiate a model

    sim = DEV_SIM(
        world_file="../worlds/robot_world.yaml", disable_plotting=False
    )  # instantiate environment

    dev_replay_buffer = get_buffer(
        deviation_model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )

    train_replay_buffer = get_buffer(
        train_model,
        sim,
        load_saved_buffer,
        pretrain,
        pretraining_iterations,
        training_iterations,
        batch_size,
    )

    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0, override_lin=0.0, override_ang=0.0
    )  # get the initial step state

    while epoch < max_epochs:  # train until max_epochs is reached
        switch = bool(episode%2)
        state, terminal = train_model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment
        action = train_model.get_action(np.array(state), True)  # get an action from the model
        dev_state, _ = deviation_model.prepare_state(latest_scan, distance, cos, sin, collision, goal, action)
        override_action = deviation_model.get_action(np.array(dev_state), ~switch)
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=action[0], ang_velocity=action[1], override_lin=override_action[0], override_ang=override_action[1], switch=switch
        )  # get data from the environment
        if switch:
            next_state, terminal = train_model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            train_replay_buffer.add(
                state, action, reward, terminal, next_state
            )
        else:
            next_state, terminal = deviation_model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            # print(len(next_state))
            dev_replay_buffer.add(
                dev_state, override_action, reward, terminal, next_state
            )

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1
            if episode % train_every_n == 0:
                train_model.train(
                    replay_buffer=train_replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )
                deviation_model.train(
                    replay_buffer=dev_replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )

            steps = 0
        else:
            steps += 1

        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            evaluate(train_model, epoch, sim, eval_episodes=nr_eval_episodes)

def compute_action(dist, sin_theta, cos_theta,
                   k_v=0.8,      # linear gain
                   k_w=1.5,      # angular gain
                   v_max=0.5,    # max linear velocity (m/s)
                   w_max=1):   # max angular velocity (rad/s)
    """
    Compute linear and angular velocity commands for a differential drive robot
    given polar coordinates of the goal.

    Args:
        dist: distance to the goal (r >= 0)
        sin_theta: sin(angle_to_goal)
        cos_theta: cos(angle_to_goal)
        k_v: gain for linear velocity
        k_w: gain for angular velocity
        v_max: max linear speed
        w_max: max angular speed

    Returns:
        v, w: linear and angular velocities
    """
    theta = math.atan2(sin_theta, cos_theta)   # in [-pi, pi]
    w = k_w * theta

    v = k_v * dist

    v *= max(0.0, cos_theta)
    v = max(-v_max, min(v, v_max))
    w = max(-w_max, min(w, w_max))

    return v, w


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            # a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=action[0], ang_velocity=action[1], override_lin=0, override_ang=0, switch=True
            )
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
