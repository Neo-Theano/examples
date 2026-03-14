# Graph Attention Network (GAT)

Multi-head attention GCN for node classification on a synthetic graph.

## Architecture

```
GATLayer(16, 8, heads=8) -> ReLU -> GATLayer(64, 4, heads=1)
```

**Total parameters:** 1,416
Graph: 50 nodes, 16 features, 4 classes, 8 attention heads

## Training

```bash
cargo run --release --bin gat
```

**Hyperparameters:** Adam optimizer, lr=0.005, CrossEntropyLoss, 30 epochs

### Training Output

```
=== Graph Attention Network (GAT) — Node Classification ===

Graph: 50 nodes, 16 features, 4 classes, 8 attention heads
Model: GATLayer(16, 8, heads=8) -> ReLU -> GATLayer(64, 4, heads=1)
Total parameters: 1416

Epoch [  1/30]  Loss: 1.6241
Epoch [ 15/30]  Loss: 1.6241
Epoch [ 30/30]  Loss: 1.6241

Model saved to gat_model.safetensors
```

## Inference

```bash
cargo run --release --bin gat-infer
```

### Inference Output

```
=== GAT Inference ===

--- Node classification on synthetic graph ---
Graph: 10 nodes, 16 features, 4 classes

  Node  0: predicted class 0 (logits: [1.852, -0.127, 0.681, 1.097])
  Node  5: predicted class 0 (logits: [2.280, 0.021, 0.778, 0.992])
  ...

Accuracy on synthetic test graph: 30.00%

--- Layer 1 attention weights (head 0, first 5 nodes) ---
  Node 0: [0:-0.0984, 8:-0.0076]
  Node 1: [1:-0.2343, 2:-0.0598, 6:0.0266, 7:-0.0900, 9:-0.1775]
  ...

Inference complete.
```

## Model

Saved to `gat_model.safetensors` using SafeTensors format.
