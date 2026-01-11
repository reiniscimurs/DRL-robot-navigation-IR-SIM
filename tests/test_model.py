from pathlib import Path

import torch

from robot_nav.SIM_ENV.marl_sim import MARL_SIM
from robot_nav.models.RCPG.RCPG import RCPG
from robot_nav.models.TD3.TD3 import TD3
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.DDPG.DDPG import DDPG
from robot_nav.utils import get_buffer
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.marl.marlTD3.marlTD3 import TD3 as marlTD3
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "model, state_dim",
    [
        (RCPG, 185),
        (CNNTD3, 185),
        (TD3, 10),
        (SAC, 10),
        (DDPG, 10),
    ],
)
def test_models(model, state_dim):
    test_model = model(
        state_dim=state_dim,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM

    prefilled_buffer = get_buffer(
        model=test_model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=[PROJECT_ROOT.joinpath("tests/test_data.yml")],
    )

    test_model.train(
        replay_buffer=prefilled_buffer,
        iterations=2,
        batch_size=8,
    )


@pytest.mark.parametrize(
    "model, state_dim",
    [
        (CNNTD3, 185),
        (TD3, 10),
        (DDPG, 10),
    ],
)
def test_max_bound_models(model, state_dim):
    test_model = model(
        state_dim=state_dim,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
        use_max_bound=True,
    )  # instantiate a model

    sim = SIM

    prefilled_buffer = get_buffer(
        model=test_model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=[PROJECT_ROOT.joinpath("tests/test_data.yml")],
    )

    test_model.train(
        replay_buffer=prefilled_buffer,
        iterations=2,
        batch_size=8,
    )


def test_marl_models():
    sim = MARL_SIM("/tests/test_marl_world.yaml", disable_plotting=True)
    test_model = marlTD3(
        state_dim=11,
        action_dim=2,
        max_action=1,
        num_robots=sim.num_robots,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    replay_buffer = get_buffer(
        model=test_model,
        sim=sim,
        load_saved_buffer=False,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
    )

    for _ in range(10):
        connections = torch.tensor(
            [[0.0 for _ in range(sim.num_robots - 1)] for _ in range(3)]
        )
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
        ) = sim.step([[0, 0] for _ in range(sim.num_robots)], connections)
        state, terminal = test_model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )
        replay_buffer.add(
            state, [[0, 0] for _ in range(sim.num_robots)], reward, terminal, state
        )

    test_model.train(
        replay_buffer=replay_buffer,
        iterations=2,
        batch_size=8,
    )
