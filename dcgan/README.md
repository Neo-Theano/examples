# DCGAN (Deep Convolutional GAN)

Generative Adversarial Network with separate generator and discriminator for generating 28x28 images.

## Architecture

```
Generator:     Linear(100, 256) -> ... -> Linear(1024, 784) -> Tanh
Discriminator: Linear(784, 512) -> ... -> Linear(256, 1) -> Sigmoid
```

**Generator parameters:** 1,486,352
**Discriminator parameters:** 533,505

## Training

```bash
cargo run --release --bin dcgan
```

**Hyperparameters:** Adam optimizer, lr=2e-4, beta1=0.5, beta2=0.999, BCELoss, 20 epochs, batch_size=64, latent_dim=100

### Training Output

```
=== Deep Convolutional GAN (DCGAN) ===

Epoch [ 1/20]  D Loss: 2.2296  G Loss: 0.1759
Epoch [10/20]  D Loss: 2.2382  G Loss: 0.1804
Epoch [20/20]  D Loss: 2.2654  G Loss: 0.1793

Generating sample from trained generator...
Sample stats: min=-0.9992, max=0.9992, shape=[1, 784]

Model saved to dcgan_model.safetensors
```

## Inference

```bash
cargo run --release --bin dcgan-infer
```

### Inference Output

```
=== DCGAN Inference ===

Model loaded from dcgan_model.safetensors

--- Generating images from random noise ---
  Sample 1: shape=[1, 784], mean=-0.0463, std=0.7519, range=[-0.9999, 0.9999]
  Sample 2: shape=[1, 784], mean=-0.0376, std=0.6788, range=[-0.9948, 0.9984]
  Sample 3: shape=[1, 784], mean=-0.0249, std=0.7779, range=[-1.0000, 0.9999]

--- Discriminator scores on generated samples ---
  Sample 1: D(G(z)) = 0.6832
  Sample 2: D(G(z)) = 0.8577
  Sample 3: D(G(z)) = 0.8206

Inference complete.
```

## Model

Saved to `dcgan_model.safetensors` using SafeTensors format.
