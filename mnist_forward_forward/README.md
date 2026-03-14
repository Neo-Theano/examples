# MNIST Forward-Forward

Implementation of Geoffrey Hinton's Forward-Forward algorithm (2022) — trains each layer independently without backpropagation using a local "goodness" objective.

## Architecture

```
Layer 0: Linear(784, 500) -> ReLU -> Normalize  [392,500 params, threshold=2.0]
Layer 1: Linear(500, 500) -> ReLU -> Normalize  [250,500 params, threshold=2.0]
Layer 2: Linear(500, 500) -> ReLU -> Normalize  [250,500 params, threshold=2.0]
```

**Total parameters:** 893,500

## Training

```bash
cargo run --release --bin mnist_forward_forward
```

**Hyperparameters:** Per-layer Adam optimizers, lr=0.001, goodness threshold=2.0, 5 epochs

### Training Output

```
Neo Theano — MNIST Forward-Forward Algorithm Example
(Hinton 2022: Training without backpropagation)

Training with Forward-Forward algorithm...
(Each layer trained independently - no backprop through the network!)

  Epoch [1/5] Average Loss: 0.8133
  Epoch [2/5] Average Loss: 0.8133
  Epoch [3/5] Average Loss: 0.8133
  Epoch [4/5] Average Loss: 0.8133
  Epoch [5/5] Average Loss: 0.8133

Evaluating...
Test Accuracy: 12.50% (5/40 correct)

Key insight: No gradient flows between layers!
Each layer is trained with its own local objective (goodness).
Model saved to mnist_ff_model.safetensors
```

## Inference

```bash
cargo run --release --bin mnist_forward_forward-infer
```

### Inference Output

```
=== MNIST Forward-Forward Inference ===

Model loaded from mnist_ff_model.safetensors
  Network has 3 layers

--- Classifying synthetic images ---
  Sample 1: Predicted digit = 8
  Sample 2: Predicted digit = 2
  Sample 3: Predicted digit = 8
  Sample 4: Predicted digit = 2
  Sample 5: Predicted digit = 2

Inference complete.
```

## Model

Saved to `mnist_ff_model.safetensors` using SafeTensors format.
