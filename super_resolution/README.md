# Super Resolution (ESPCN)

Efficient Sub-Pixel Convolutional Network for 2x image upscaling.

## Architecture

```
ESPCN: Conv layers for feature extraction + sub-pixel shuffle for 2x upscaling
Input:  8x8 -> Output: 16x16
```

**Total parameters:** 21,284

## Training

```bash
cargo run --release --bin super_resolution
```

**Hyperparameters:** Adam optimizer, lr=1e-3, MSELoss, 20 epochs, batch_size=4

### Training Output

```
=== Super-Resolution (ESPCN) ===

Upscale factor: 2x
Input size: 8x8 -> Output size: 16x16

Epoch [ 1/20]  MSE Loss: 0.413343  PSNR: 3.84 dB
Epoch [ 7/20]  MSE Loss: 0.396129  PSNR: 4.02 dB
Epoch [17/20]  MSE Loss: 0.396258  PSNR: 4.02 dB
Epoch [20/20]  MSE Loss: 0.407147  PSNR: 3.90 dB

Testing on a single image...
Test MSE: 0.434929  PSNR: 3.62 dB
Input shape:  [1, 1, 8, 8]
Output shape: [1, 1, 16, 16]

Model saved to super_resolution_model.safetensors
```

## Inference

```bash
cargo run --release --bin super-resolution-infer
```

### Inference Output

```
=== Super Resolution Inference ===

--- Output Pixel Statistics ---
  Mean:  0.1029
  Std:   0.3861
  Min:   -1.4869
  Max:   1.5596
  Total pixels: 256

--- First 10 output pixels ---
  pixel[0] = -0.0861
  pixel[1] = 0.0585
  pixel[2] = 0.1760
  ...

Inference complete.
```

## Model

Saved to `super_resolution_model.safetensors` using SafeTensors format.
