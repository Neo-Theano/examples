# ImageNet (ResNet-18)

ResNet-18 for 1000-class ImageNet-style classification on synthetic 224x224 RGB images.

## Architecture

```
ResNet-18:
  Conv2d(3, 64, 7, stride=2) -> BatchNorm -> ReLU -> MaxPool(3, stride=2)
  Layer1: 2x BasicBlock(64)
  Layer2: 2x BasicBlock(128, stride=2)
  Layer3: 2x BasicBlock(256, stride=2)
  Layer4: 2x BasicBlock(512, stride=2)
  AdaptiveAvgPool2d(1,1) -> Linear(512, 1000)
```

**Total parameters:** 11,689,512

## Training

```bash
cargo run --release --bin imagenet
```

**Hyperparameters:** SGD optimizer, lr=0.01, momentum=0.9, weight_decay=1e-4, CrossEntropyLoss, 3 epochs, batch_size=2

### Training Output

```
=== ResNet-18 ImageNet Training (Synthetic Data) ===

Model parameters: 11689512
Epoch [1/3]  Loss: 7.7305  Top-1 Accuracy: 0.00%
Epoch [2/3]  Loss: 8.0371  Top-1 Accuracy: 0.00%
Epoch [3/3]  Loss: 7.3934  Top-1 Accuracy: 0.00%

Model saved to resnet18_model.safetensors (93521900 bytes)
```

## Inference

```bash
cargo run --release --bin imagenet-infer
```

### Inference Output

```
=== ResNet-18 Inference ===

Loaded state dict with 62 tensors
ResNet-18 model reconstructed.

Top-5 predictions:
  #1: class 691 (score: 3.8731)
  #2: class 815 (score: 3.4142)
  #3: class 594 (score: 3.3114)
  #4: class 305 (score: 3.1941)
  #5: class 805 (score: 3.1745)

Inference complete.
```

## Model

Saved to `resnet18_model.safetensors` using SafeTensors format (~89 MB).
