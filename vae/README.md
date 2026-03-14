# Variational Autoencoder (VAE)

Standard VAE with reconstruction loss (BCE) and KL divergence regularization.

## Architecture

```
Encoder: Linear(784, 400) -> ReLU -> [Linear(400, 20), Linear(400, 20)]  (mu, logvar)
Reparameterize: z = mu + std * epsilon
Decoder: Linear(20, 400) -> ReLU -> Linear(400, 784) -> Sigmoid
```

**Total parameters:** 652,824

## Training

```bash
cargo run --release --bin vae
```

**Hyperparameters:** Adam optimizer, lr=1e-3, BCE + KL loss, 20 epochs, batch_size=64

### Training Output

```
=== Variational Autoencoder (VAE) ===

Epoch [ 1/20]  Total Loss: 53226.26  Recon Loss: 52560.83  KL Divergence: 665.42
Epoch [10/20]  Total Loss: 53764.94  Recon Loss: 53095.16  KL Divergence: 669.79
Epoch [20/20]  Total Loss: 53608.97  Recon Loss: 52943.25  KL Divergence: 665.72

Model saved to vae_model.safetensors
```

## Inference

```bash
cargo run --release --bin vae-infer
```

### Inference Output

```
=== VAE Inference ===

Model loaded from vae_model.safetensors

--- Encoding a sample image ---
Latent mu (first 5):    [0.098, 0.900, -1.136, 1.576, -0.016]
Latent logvar (first 5): [-0.330, -0.295, -0.558, -1.170, 0.346]
Reconstruction (first 10 pixels): [0.958, 0.807, 0.312, 0.853, 0.115, ...]

--- Generating from latent space ---
  Sample 1: mean pixel=0.4782, std=0.2392, range=[0.0299, 0.9889]
  Sample 2: mean pixel=0.4702, std=0.2460, range=[0.0079, 0.9852]
  Sample 3: mean pixel=0.4809, std=0.2688, range=[0.0130, 0.9959]

Inference complete.
```

## Model

Saved to `vae_model.safetensors` using SafeTensors format.
