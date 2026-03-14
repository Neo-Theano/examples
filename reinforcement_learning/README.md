# Reinforcement Learning (REINFORCE)

REINFORCE policy gradient algorithm for a multi-armed bandit environment.

## Architecture

```
Linear(4, 128) -> ReLU -> Linear(128, 5) -> Softmax
```

**Total parameters:** 4 (policy network)

## Training

```bash
cargo run --release --bin reinforcement_learning
```

**Hyperparameters:** Adam optimizer, gamma=0.99, max_steps=20, 10 episodes per batch, 20 batches

### Training Output

```
Reinforcement Learning — REINFORCE (Policy Gradient) Example
State dim: 4, Actions: 5, Hidden: 128
Gamma: 0.99, Max steps: 20, Episodes per batch: 10

Batch [  1/20] — Avg Reward:  -2.965, Avg Loss:   1.319
Batch [  5/20] — Avg Reward:  -1.237, Avg Loss:   2.702
Batch [ 10/20] — Avg Reward:   1.132, Avg Loss:   1.092
Batch [ 15/20] — Avg Reward:  -1.077, Avg Loss:   1.139
Batch [ 20/20] — Avg Reward:  -0.398, Avg Loss:  -0.468

Final evaluation (5 episodes):
  Episode 1 — Reward:  -2.179
  Episode 2 — Reward:  -5.522
  Episode 3 — Reward:   2.243
  Episode 4 — Reward:   6.265
  Episode 5 — Reward:  -6.817

Model saved to rl_model.safetensors
```

## Inference

```bash
cargo run --release --bin reinforcement-learning-infer
```

### Inference Output

```
=== REINFORCE Policy Inference ===

Model loaded from rl_model.safetensors
State dim: 4, Num actions: 5

--- Action Probabilities for Sample States ---
  State 1: [1.0, 0.0, 0.0, 0.0] -> probs: [0.0446, 0.4472, 0.1591, 0.1461, 0.2030] -> best action: 1
  State 2: [0.0, 1.0, 0.0, 0.0] -> probs: [0.0717, 0.3501, 0.0973, 0.2031, 0.2777] -> best action: 1
  State 3: [0.0, 0.0, 1.0, 0.0] -> probs: [0.1194, 0.2961, 0.1235, 0.1030, 0.3580] -> best action: 4
  State 4: [0.0, 0.0, 0.0, 1.0] -> probs: [0.0769, 0.2792, 0.3117, 0.1177, 0.2145] -> best action: 2
  State 5: [0.5, 0.5, -0.5, -0.5] -> probs: [0.0793, 0.3321, 0.1915, 0.1926, 0.2046] -> best action: 1

Inference complete.
```

## Model

Saved to `rl_model.safetensors` using SafeTensors format.
