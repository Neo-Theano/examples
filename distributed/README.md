# Distributed Data Parallel (DDP) Training

Demonstrates the DistributedDataParallel API with parameter broadcasting, gradient synchronization via all-reduce, and collective operations.

## Architecture

```
SimpleModel:
  Linear(128, 64) -> ReLU -> Linear(64, 10)
```

**Total parameters:** 67,210

## Training

```bash
cargo run --release --bin distributed
```

**Hyperparameters:** SGD optimizer, lr=0.01, momentum=0.9, CrossEntropyLoss, 5 epochs, rank=0, world_size=1, backend=Gloo

### Training Output

```
=== Distributed Data Parallel (DDP) Training Example ===

Process group initialised: rank=0, world_size=1, backend=Gloo
Model parameters: 67210
DDP wrapper created (broadcast_buffers=true, bucket_size=25MB)

Broadcasting initial parameters from rank 0...
Parameters synchronised across 1 processes.

--- Training ---
Epoch [1/5]  Loss: 3.0264  Grad Norm: 4.5485  [synchronised]
Epoch [2/5]  Loss: 2.3614  Grad Norm: 2.6620  [synchronised]
Epoch [3/5]  Loss: 2.4811  Grad Norm: 3.1586  [synchronised]
Epoch [4/5]  Loss: 2.7810  Grad Norm: 3.7825  [synchronised]
Epoch [5/5]  Loss: 2.7380  Grad Norm: 4.1885  [synchronised]

--- Collective Operations Demo ---
all_reduce(Sum):   input=[1.0, 2.0, 3.0]  output=[1.0, 2.0, 3.0]
broadcast(rank=0): input=[1.0, 2.0, 3.0]  output=[1.0, 2.0, 3.0]
barrier():         all ranks synchronised

Model saved to distributed_model.safetensors
```

## Inference

```bash
cargo run --release --bin distributed-infer
```

### Inference Output

```
=== Distributed Model Inference ===

Model loaded from distributed_model.safetensors

Input shape: [1, 128]
Output logits (10 classes):
  Class 0: 1.4616
  Class 1: -0.6161
  Class 2: 1.1331
  Class 3: 0.8458
  Class 4: 0.6980
  Class 5: -0.2692
  Class 6: -0.3949
  Class 7: -0.8444
  Class 8: -0.4374
  Class 9: 0.1837

Predicted class: 0 (logit: 1.4616)

Inference complete.
```

## Model

Saved to `distributed_model.safetensors` using SafeTensors format.
