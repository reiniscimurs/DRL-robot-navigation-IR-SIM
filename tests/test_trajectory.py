import numpy as np
import pytest

from robot_nav.SIM_ENV.trajectory import SplineTrajectory


def test_straight_projection_and_arc_lookahead():
    path = SplineTrajectory([[0, 0], [10, 0]])
    point, distance = path.project([3, 2])
    np.testing.assert_allclose(point, [3, 0], atol=1e-6)
    assert distance == pytest.approx(3)
    np.testing.assert_allclose(path.lookahead_points(distance, [0, 2, 20]),
                               [[3, 0], [5, 0], [10, 0]], atol=1e-6)
    np.testing.assert_allclose(path.project([-2, 1])[0], [0, 0])
    np.testing.assert_allclose(path.project([12, 1])[0], [10, 0])


def test_curve_interpolates_waypoints_and_projects_normally():
    path = SplineTrajectory([[0, 0], [2, 3], [6, 2], [8, 5]])
    for waypoint in [[0, 0], [2, 3], [6, 2], [8, 5]]:
        point, _ = path.project(waypoint)
        np.testing.assert_allclose(point, waypoint, atol=1e-6)
    distance = path.length * 0.4
    point = path.point_at(distance)
    heading = path.heading_at(distance)
    query = point + 0.1 * np.array([-np.sin(heading), np.cos(heading)])
    projected, progress = path.project(query)
    np.testing.assert_allclose(projected, point, atol=1e-5)
    assert progress == pytest.approx(distance, abs=1e-5)
    # Check arc offsets against a much denser, independent chord sum.
    dense = path.point_at(np.linspace(distance, distance + 1, 10001))
    assert np.linalg.norm(np.diff(dense, axis=0), axis=1).sum() == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize("points", [[], [[1, 2]], [[0, 0], [0, 0]],
                                     [[0, 0], [np.nan, 1]], [[0], [1]]])
def test_invalid_waypoints(points):
    with pytest.raises(ValueError):
        SplineTrajectory(points)


def test_lookahead_validation_and_empty_points():
    path = SplineTrajectory([[0, 0], [1, 1]])
    assert path.lookahead_points(0, []).shape == (0, 2)
    for offsets in [[-1], [np.inf], [[1]]]:
        with pytest.raises(ValueError):
            path.lookahead_points(0, offsets)


def test_sim_uses_tangent_and_robot_frame():
    from robot_nav.SIM_ENV.line_sim import SIM

    class Environment:
        disable_all_plot = True

        def step(self, **kwargs):
            pass

        def render(self):
            pass

        def get_robot_state(self):
            return np.array([[0.0], [2.0], [np.pi / 2]])

    sim = SIM.__new__(SIM)
    sim.env = Environment()
    sim.trajectory = SplineTrajectory([[0, 0], [0, 10]])
    sim.lookahead_distances = np.array([1, 3])
    state, _, _ = sim.step(0, 0)
    assert len(state) == 10
    assert state[3] == pytest.approx(0)
    np.testing.assert_allclose(state[6:], [1, 0, 3, 0], atol=1e-6)
    sim.trajectory = None
    assert len(sim.step(0, 0)[0]) == 6


def test_irsim_trajectory_overlay_refreshes():
    # Exercise IRSim's real Matplotlib artists without opening a GUI window.
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path
    from robot_nav.SIM_ENV.line_sim import SIM

    world = Path(__file__).resolve().parents[1] / "robot_nav/worlds/line_world.yaml"
    sim = SIM(str(world))
    sim.env.display = False
    try:
        for _ in range(3):
            sim.step(0, 0)
            plot = sim.env._env_plot
            assert len(plot.dyna_line_list) == 1
            assert len(plot.dyna_point_list) == 2
            line = plot.dyna_line_list[0][0]
            np.testing.assert_allclose(line.get_xydata(), sim.trajectory.sample_points())
            np.testing.assert_allclose(plot.dyna_point_list[1].get_offsets(),
                                       sim.lookahead_points)
        sim.reset(robot_state=[[5], [15], [0]])
        assert len(sim.env._env_plot.dyna_line_list) == 1
        assert len(sim.env._env_plot.dyna_point_list) == 2
        # No extra artists on physics steps between rendered frames.
        sim.env._world.sample_time = sim.env.step_time * 3
        for _ in range(6):
            sim.step(0, 0)
            assert len(sim.env._env_plot.dyna_line_list) == 1
            assert len(sim.env._env_plot.dyna_point_list) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(sim.env._env_plot.fig)


def test_world_yaml_configures_training_trajectory():
    from pathlib import Path
    from robot_nav.SIM_ENV.line_sim import SIM

    world = Path(__file__).resolve().parents[1] / "robot_nav/worlds/line_world.yaml"
    sim = SIM(str(world), disable_plotting=True)
    assert sim.trajectory is not None
    assert sim.state_dim == 12
    np.testing.assert_allclose(sim.lookahead_distances, [1, 3, 5])
    state, _, _ = sim.step(0, 0)
    assert len(state) == sim.state_dim
    np.testing.assert_allclose(sim.trajectory.point_at(0), [5, 15])
    assert sim.trajectory.point_at(sim.trajectory.length * 0.4)[1] > 15
