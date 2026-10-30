Run multi-robot line following from the repository root:

```bash
python -m robot_nav.marl_line_train --headless
```

The existing single-robot entry point `robot_nav.rl_line_train` remains available.
Edit `worlds/marl_line_world.yaml` to set the robot count and one reference path
per robot. References use the same `SplineTrajectory` interpolation and
lookahead distances as the single-robot trainer. Interior waypoints and initial
poses vary on training resets; evaluation uses the configured paths and poses.

The shared MARL TD3 actor produces all robot actions in one call. Both actor and
critic exchange peer information only through the existing IGA geometry and
attention edges. Each policy head receives its own encoding plus its IGA output;
there is no flattened global-state or joint-action input to the policy or critic.

Each observation is `[x, y, cos(theta), sin(theta), distance, cos(projection_angle),
sin(projection_angle), linear_velocity, angular_velocity, tangent_heading_error,
lookahead_forward_1, lookahead_left_1, ..., target_x, target_y]`.
The target is three arc units ahead and occupies IGA's existing goal fields.
The local encoder includes heading error and lookahead points; peer messages
retain IGA's existing pose, velocity and target geometry. Original 11-feature
IGA observations and checkpoints keep their original encoder dimensions.
Line-following checkpoints have a larger input encoder and are saved as
`IGA-line` under the existing MARL checkpoint directory.

`JointReplayBuffer` stores entire synchronized scene transitions. Rewards use
the original tracking formula with a 20-point collision penalty. Any collision,
path completion (within 0.3 of the endpoint), or departure from world bounds
ends the scene for all robots and marks all agents terminal. Step limits truncate
the episode and retain bootstrapping. Collision mode is unobstructed so IRSim
reports collisions while this wrapper controls episode termination.

Evaluation disables exploration and stochastic IGA edge selection. Training
keeps IGA Gumbel sampling with learned connections. Metrics and
checkpoint cadence follow MARL TD3; a final checkpoint is saved on completion.
Useful quick-run options include `--epochs`, `--episodes-per-epoch`,
`--max-steps`, `--training-iterations`, `--batch-size`, and `--eval-episodes`.

Connection learning uses no BCE or distance-based labels. During training every
peer edge is evaluated and its message is multiplied by its Gumbel gate after
softmax, without renormalization. Evaluation thresholds gates at 0.5 while
preserving that attention normalization. Displayed connection strengths include
the gate. Selector logits returned by IGA are open-minus-closed log-odds.

The actor minimizes `-Q + communication_weight * mean_peer_probability
- entropy_weight * entropy`. The critic minimizes TD error minus the entropy
bonus and has no communication cost. Entropy includes binary selector entropy
and soft attention entropy over candidate peers. Defaults ramp actor cost from
zero to 0.01 over 1,000 gradient steps and decay entropy from 0.01 to zero over
10,000 steps. `TD3.train` exposes `communication_weight`, `communication_warmup`,
`entropy_weight`, `entropy_decay_steps`, and `final_entropy_weight`. Schedules
persist across training calls within a run; loading weights starts them afresh.
These starting values require validation in a full training run.

Displayed training and evaluation show connection lines using the actor's actual
`gate * soft_attention` weights. Thicker, more opaque lines indicate stronger
connections; the color identifies the receiving robot. Opposite directions are
slightly offset to keep both visible. Closed connections are hidden, and the
lines refresh alongside the existing projection and lookahead markers.
