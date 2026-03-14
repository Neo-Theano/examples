# Siamese Network

Siamese network for metric learning with contrastive loss, producing 64-dimensional embeddings.

## Architecture

```
Shared branch (applied to both inputs):
  Linear(784, 256) -> ReLU
  Linear(256, 128) -> ReLU
  Linear(128, 64)

Contrastive Loss: margin=2.0
```

**Total parameters:** 242,112

## Training

```bash
cargo run --release --bin siamese_network
```

**Hyperparameters:** Adam optimizer, lr=1e-3, contrastive loss, margin=2.0, 20 epochs, batch_size=32

### Training Output

```
=== Siamese Network ===

Embedding dimension: 64
Contrastive margin: 2

Epoch [ 1/20]  Loss: 8.0604  Avg Same Dist: 4.0102  Avg Diff Dist: 5.2437
Epoch [10/20]  Loss: 7.9970  Avg Same Dist: 3.9373  Avg Diff Dist: 5.2052
Epoch [20/20]  Loss: 7.9607  Avg Same Dist: 4.0232  Avg Diff Dist: 5.1662

Testing with sample pairs...
  Pair 1: label=same, distance=4.5245
  Pair 2: label=diff, distance=5.0153
  Pair 3: label=same, distance=4.1708

Model saved to siamese_model.safetensors
```

## Inference

```bash
cargo run --release --bin siamese_network-infer
```

### Inference Output

```
=== Siamese Network Inference ===

Model loaded from siamese_model.safetensors

Embedding 1 (first 5): [-0.3152, -1.0075, 1.2185, 0.8879, 0.1915]
Embedding 2 (first 5): [-0.1951, -0.5801, 0.8227, 0.5150, 0.1496]

Euclidean distance: 3.4987
Distance squared:   12.2407

--- Testing with similar image (noisy copy) ---
Distance (original vs noisy copy): 0.4001

Inference complete.
```

## Model

Saved to `siamese_model.safetensors` using SafeTensors format.
