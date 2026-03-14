# Vision Transformer (ViT)

Vision Transformer for CIFAR-like image classification, splitting images into patches and processing them with a Transformer encoder.

## Architecture

```
Image:    32x32 RGB -> 4x4 patches -> 64 patches
Embed:    Linear(48, 64) + learnable position embeddings
Encoder:  4x TransformerBlock(embed_dim=64, num_heads=4)
Classify: LayerNorm -> Linear(64, 10)
```

**Total parameters:** 208,074

## Training

```bash
cargo run --release --bin vision_transformer
```

**Hyperparameters:** Adam optimizer, lr=0.001, CrossEntropyLoss, 5 epochs, batch_size=4

### Training Output

```
=== Vision Transformer (ViT) Training (Synthetic Data) ===

ViT Configuration:
  Image size:   32x32
  Patch size:   4x4
  Num patches:  64
  Embed dim:    64
  Num heads:    4
  Num blocks:   4
  Num classes:  10
  Total params: 208074

Epoch [1/5]  Loss: 3.5468
Epoch [2/5]  Loss: 2.8480
Epoch [3/5]  Loss: 3.9697
Epoch [4/5]  Loss: 4.3096
Epoch [5/5]  Loss: 3.6277

Model saved to vit_model.safetensors (1671216 bytes)
```

## Inference

```bash
cargo run --release --bin vision_transformer-infer
```

### Inference Output

```
=== Vision Transformer (ViT) Inference ===

Loaded state dict with 72 tensors
ViT model reconstructed.

Top-5 predictions:
  #1: class 1 (score: 3.2949)
  #2: class 5 (score: 2.1579)
  #3: class 6 (score: 1.6061)
  #4: class 2 (score: 1.0796)
  #5: class 4 (score: 1.0557)

Inference complete.
```

## Model

Saved to `vit_model.safetensors` using SafeTensors format.
