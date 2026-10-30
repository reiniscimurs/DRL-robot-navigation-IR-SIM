import numpy as np
import pytest
import torch

from robot_nav.models.MARL.Attention.iga import Attention
from robot_nav.replay_buffer import JointReplayBuffer
from robot_nav.SIM_ENV.marl_line_sim import MARL_LINE_SIM


def test_joint_replay_keeps_scene_alignment_and_copies():
    replay = JointReplayBuffer(3)
    states = np.arange(54).reshape(3, 18)
    replay.add(states, np.zeros((3, 2)), np.arange(3), np.zeros(3), states + 1)
    states[:] = -1
    s, a, r, done, next_s = replay.sample_batch(8)
    assert s.shape == (1, 3, 18)
    np.testing.assert_array_equal(next_s - s, 1)
    np.testing.assert_array_equal(r[0], np.arange(3))
    with pytest.raises(ValueError):
        replay.add(states, np.zeros((3, 2)), 1, False, states)


def test_disabled_iga_edges_isolate_local_policy_features():
    attention = Attention(16, local_state_dim=12).eval()
    with torch.no_grad():
        attention.hard_encoding.weight.zero_()
        attention.hard_encoding.bias.copy_(torch.tensor([100., -100.]))
        observation = torch.randn(1, 3, 18)
        original = attention(observation)[0]
        changed = observation.clone()
        changed[:, 1:] += 10
        updated = attention(changed)[0]
    torch.testing.assert_close(original[0], updated[0])
    assert not torch.allclose(original[1:], updated[1:])


def test_scene_reset_step_and_collision():
    sim = MARL_LINE_SIM(disable_plotting=True)
    try:
        state = sim.reset(randomize=False)
        assert state.shape == (3, sim.state_dim)
        for i, path in enumerate(sim.trajectories):
            np.testing.assert_allclose(state[i, :2], path.point_at(0))
            assert state[i, 9] == pytest.approx(0, abs=1e-6)
        before = [robot.state.copy() for robot in sim.env.robot_list]
        next_state, rewards, dones = sim.step(np.array([[0.5, 0]] * 3))
        for previous, robot in zip(before, sim.env.robot_list):
            assert np.linalg.norm(robot.state[:2] - previous[:2]) == pytest.approx(0.15)
        assert np.isfinite(next_state).all() and np.isfinite(rewards).all()
        assert not dones.any()
        sim.env.robot_list[1].set_state(sim.env.robot_list[0].state.copy())
        _, rewards, dones = sim.step(np.zeros((3, 2)))
        assert dones[:2].all()
        assert (rewards[:2] < -10).all()
    finally:
        sim.env.end()


def test_iga_td3_updates_with_partial_joint_batch(tmp_path):
    from robot_nav.models.MARL.marlTD3.marlTD3 import TD3

    sim = MARL_LINE_SIM(disable_plotting=True)
    model = TD3(sim.state_dim, 2, 1, torch.device('cpu'), sim.num_robots,
                attention='iga', save_directory=tmp_path)
    try:
        state = sim.reset(randomize=False)
        actions, _, _ = model.get_action(state, False)
        assert actions.shape == (3, 2)
        next_state, rewards, done = sim.step(np.zeros((3, 2)))
        replay = JointReplayBuffer(4)
        replay.add(state, actions, rewards, done, next_state)
        before = model.actor.policy_head[0].weight.detach().clone()
        model.train(replay, iterations=1, batch_size=8)
        assert not torch.equal(before, model.actor.policy_head[0].weight)
        assert all(torch.isfinite(p).all() for p in model.actor.parameters())
    finally:
        model.writer.close()
        sim.env.end()


def test_selector_entropy_has_gradients_without_selected_connections():
    attention = Attention(16).eval()
    with torch.no_grad():
        attention.hard_encoding.weight.zero_()
        attention.hard_encoding.bias.copy_(torch.tensor([2., -2.]))
    _, _, _, entropy, weights, _ = attention(torch.randn(2, 2, 11))
    assert (weights < 0.5).all()  # No messages: only selector entropy contributes.
    expected = torch.distributions.Categorical(logits=torch.tensor([2., -2.])).entropy()
    torch.testing.assert_close(entropy, expected)
    (-entropy).backward()
    grad = attention.hard_encoding.bias.grad
    assert torch.isfinite(grad).all()
    # Gradient descent on -entropy moves the two logits toward each other.
    assert grad[0] > 0 and grad[1] < 0
    assert attention.hard_encoding.weight.grad.abs().sum() > 0


def test_soft_attention_entropy_retains_gradients_and_is_not_degree_scaled():
    from robot_nav.models.MARL.Attention.iga import GoalAttentionLayer, entropy_from_attention

    torch.manual_seed(10)
    layer = GoalAttentionLayer(8, 10, 8)
    edges = torch.tensor([[1, 2], [0, 0]])
    _, probabilities = layer(torch.randn(3, 8), edges, torch.randn(2, 10))
    entropy = entropy_from_attention(probabilities, edges[1], 3)
    torch.testing.assert_close(entropy, -(probabilities * (probabilities + 1e-10).log()).sum())
    (-entropy).backward()
    grad = layer.attn_score_layer[-1].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_single_robot_entropy_is_finite_zero():
    attention = Attention(16)
    entropy = attention(torch.randn(1, 1, 11))[3]
    assert entropy.item() == 0
    entropy.backward()
    assert torch.isfinite(attention.hard_encoding.bias.grad).all()


def test_task_gradient_reaches_selector_without_auxiliary_losses():
    from robot_nav.models.MARL.marlTD3.marlTD3 import Actor, Critic

    torch.manual_seed(31)
    actor = Actor(2, 16, 'iga')
    critic = Critic(2, 16, 'iga')
    state = torch.randn(2, 3, 11)
    actions = actor(state)[0]
    q1, q2, *_ = critic(state, actions)
    (-q1.mean() + q2.square().mean()).backward()
    for network in (actor, critic):
        grad = network.attention.hard_encoding.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_message_gate_scales_after_softmax_without_renormalization():
    from robot_nav.models.MARL.Attention.iga import GoalAttentionLayer

    layer = GoalAttentionLayer(8, 10, 8)
    nodes, attributes = torch.randn(3, 8), torch.randn(2, 10)
    edges = torch.tensor([[1, 2], [0, 0]])
    full = layer(nodes, edges, attributes, torch.ones(2))[0]
    half = layer(nodes, edges, attributes, torch.full((2,), 0.5))[0]
    closed = layer(nodes, edges, attributes, torch.zeros(2))[0]
    torch.testing.assert_close(half, full * 0.5)
    torch.testing.assert_close(closed, torch.zeros_like(full))


def test_connection_cost_reduces_probability():
    attention = Attention(16)
    logits = attention(torch.randn(2, 3, 11))[1].view(2, 3, 3)
    mask = ~torch.eye(3, dtype=torch.bool)
    cost = logits.sigmoid()[:, mask].mean()
    cost.backward()
    grad = attention.hard_encoding.bias.grad
    assert grad[1] > 0 and grad[0] < 0
