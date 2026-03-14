# Fast Neural Style Transfer

Neural style transfer with a TransformerNet and a pre-trained feature extractor, combining content loss and style loss (Gram matrices).

## Architecture

```
Encoder:  Conv2d(3,32,3) -> Conv2d(32,64,3,stride=2) -> Conv2d(64,128,3,stride=2)
Residual: 2x ResidualBlock(128)
Decoder:  Conv2d(128,64,3) -> Conv2d(64,32,3) -> Conv2d(32,3,3)

Feature Extractor (VGG-like):
  Conv2d(3,16,3) -> ReLU -> Conv2d(16,32,3) -> ReLU
```

**Transformer parameters:** 776,707
**Feature extractor parameters:** 5,088

## Training

```bash
cargo run --release --bin fast_neural_style
```

**Hyperparameters:** Adam optimizer, lr=1e-3, content_weight=1.0, style_weight=1e5, 15 epochs, 16x16 images

### Training Output

```
=== Fast Neural Style Transfer ===

Epoch [ 1/15]  Total: 24243.44  Content: 0.552  Style: 0.242
Epoch [ 7/15]  Total: 21664.86  Content: 0.529  Style: 0.217
Epoch [15/15]  Total: 20652.92  Content: 0.505  Style: 0.207

Running single forward pass...
Input shape:  [1, 3, 16, 16]
Output shape: [1, 3, 4, 4]

Model saved to style_model.safetensors
```

## Inference

```bash
cargo run --release --bin fast_neural_style-infer
```

### Inference Output

```
=== Fast Neural Style Transfer Inference ===

Model loaded from style_model.safetensors
Input shape:  [1, 3, 16, 16]
Output shape: [1, 3, 4, 4]

Output statistics:
  Min:  -1.8932
  Max:  0.5927
  Mean: -0.7575
  Std:  0.5970

Per-channel statistics:
  Channel 0 (R): mean=-0.6784, min=-1.8932, max=0.5927
  Channel 1 (G): mean=-0.7567, min=-1.7656, max=-0.0116
  Channel 2 (B): mean=-0.8375, min=-1.6984, max=0.3586

Inference complete.
```

## Model

Saved to `style_model.safetensors` using SafeTensors format.
