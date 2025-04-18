from robot_nav.sim import SIM_ENV
import numpy as np


def test_sim():
    sim = SIM_ENV("/tests/test_world.yaml")
    robot_state = sim.env.get_robot_state()
    state = sim.step(1, 0)
    next_robot_state = sim.env.get_robot_state()
    assert np.isclose(robot_state[0], next_robot_state[0] - 1)
    assert np.isclose(robot_state[1], robot_state[1])

    assert len(state[0]) == 180
    assert len(sim.env.obstacle_list) == 7

    sim.reset()
    new_robot_state = sim.env.get_robot_state()
    assert np.not_equal(robot_state[0], new_robot_state[0])
    assert np.not_equal(robot_state[1], new_robot_state[1])


def test_sincos():
    sim = SIM_ENV("/tests/test_world.yaml")
    cos, sin = sim.cossin([1, 0], [0, 1])
    assert np.isclose(cos, 0)
    assert np.isclose(sin, 1)
