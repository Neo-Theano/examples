# Graph Convolutional Network (GCN)

2-layer GCN for node classification on a synthetic random graph.

## Architecture

```
GCNLayer(16, 16) -> ReLU -> GCNLayer(16, 5)
```

Graph: 100 nodes, 16 features, 5 classes, edge_prob=0.1

## Training

```bash
cargo run --release --bin gcn
```

**Hyperparameters:** Adam optimizer, lr=0.01, CrossEntropyLoss, 50 epochs

### Training Output

```
=== Graph Convolutional Network (GCN) — Node Classification ===

Graph: 100 nodes, 16 features, 5 classes
Model: GCNLayer(16, 16) -> ReLU -> GCNLayer(16, 5)

Epoch [  1/50]  Loss: 2.7712  Accuracy: 25.00%
Epoch [ 25/50]  Loss: 2.7712  Accuracy: 25.00%
Epoch [ 50/50]  Loss: 2.7712  Accuracy: 25.00%

Model saved to gcn_model.safetensors
```

## Inference

```bash
cargo run --release --bin gcn-infer
```

### Inference Output

```
=== GCN Inference ===

Model loaded from gcn_model.safetensors

--- Node classification on synthetic graph ---
Graph: 20 nodes, 16 features

  Node  0: predicted class 0 (logits: [2.169, -0.483, -0.267, -1.670, 1.427])
  Node  1: predicted class 0 (logits: [2.730, -0.915, -0.365, -1.823, 2.010])
  ...

Accuracy on synthetic test graph: 15.00%

Inference complete.
```

## Model

Saved to `gcn_model.safetensors` using SafeTensors format.
