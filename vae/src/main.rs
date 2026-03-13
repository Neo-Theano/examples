//! Variational Autoencoder (VAE) example.
//!
//! Implements a VAE with:
//! - Encoder: Linear(784, 400) -> ReLU -> (mu, logvar) heads
//! - Reparameterization trick: z = mu + std * eps
//! - Decoder: Linear(20, 400) -> ReLU -> Linear(400, 784) -> Sigmoid
//! - Loss: BCE reconstruction + KL divergence
//!
//! Trained on synthetic flattened 28x28 images.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Encoder {
    fc1: Linear,
    fc_mu: Linear,
    fc_logvar: Linear,
}

impl Encoder {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 400),
            fc_mu: Linear::new(400, 20),
            fc_logvar: Linear::new(400, 20),
        }
    }

    fn forward(&self, x: &Variable) -> (Variable, Variable) {
        let h = self.fc1.forward(x).relu().unwrap();
        let mu = self.fc_mu.forward(&h);
        let logvar = self.fc_logvar.forward(&h);
        (mu, logvar)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc_mu.parameters());
        params.extend(self.fc_logvar.parameters());
        params
    }
}

struct Decoder {
    fc1: Linear,
    fc2: Linear,
}

impl Decoder {
    fn new() -> Self {
        Self {
            fc1: Linear::new(20, 400),
            fc2: Linear::new(400, 784),
        }
    }

    fn forward(&self, z: &Variable) -> Variable {
        let h = self.fc1.forward(z).relu().unwrap();
        self.fc2.forward(&h).sigmoid().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

struct VAE {
    encoder: Encoder,
    decoder: Decoder,
}

impl VAE {
    fn new() -> Self {
        Self {
            encoder: Encoder::new(),
            decoder: Decoder::new(),
        }
    }

    /// Forward pass with reparameterization trick.
    /// Returns (reconstruction, mu, logvar).
    fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let (mu, logvar) = self.encoder.forward(x);

        // Reparameterization trick: z = mu + std * eps
        let std = logvar.mul_scalar(0.5).unwrap().exp().unwrap();
        let eps = random_normal_like(&std);
        let z = mu.add(&std.mul(&eps).unwrap()).unwrap();

        let recon = self.decoder.forward(&z);
        (recon, mu, logvar)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

/// BCE reconstruction loss (computed element-wise for numerical stability).
fn bce_reconstruction_loss(recon: &Variable, target: &Variable) -> Variable {
    let eps = 1e-8;
    let recon_data = recon.tensor().to_vec_f64().unwrap();
    let target_data = target.tensor().to_vec_f64().unwrap();

    let bce: Vec<f64> = recon_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .collect();

    let bce_tensor = Variable::new(Tensor::from_slice(&bce, recon.tensor().shape()));
    bce_tensor.sum().unwrap()
}

/// KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
fn kl_divergence(mu: &Variable, logvar: &Variable) -> Variable {
    let mu_data = mu.tensor().to_vec_f64().unwrap();
    let logvar_data = logvar.tensor().to_vec_f64().unwrap();

    let kl: Vec<f64> = mu_data
        .iter()
        .zip(logvar_data.iter())
        .map(|(&m, &lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
        .collect();

    let kl_tensor = Variable::new(Tensor::from_slice(&kl, mu.tensor().shape()));
    kl_tensor.sum().unwrap()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a Variable of standard normal random values with same shape.
fn random_normal_like(v: &Variable) -> Variable {
    let shape = v.tensor().shape();
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let data: Vec<f64> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Variable::new(Tensor::from_slice(&data, shape))
}

/// Generate a batch of synthetic flattened 28x28 images with pixel values in [0, 1].
fn synthetic_batch(batch_size: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let numel = batch_size * 784;
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 784]))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Variational Autoencoder (VAE) ===");
    println!();

    let vae = VAE::new();
    let mut optimizer = Adam::new(vae.parameters(), 1e-3);

    let batch_size = 64;
    let num_epochs = 20;
    let batches_per_epoch = 10;

    println!(
        "Model parameters: {}",
        vae.parameters().iter().map(|p| p.tensor().numel()).sum::<usize>()
    );
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", num_epochs);
    println!();

    for epoch in 1..=num_epochs {
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            let x = synthetic_batch(batch_size);
            let (recon, mu, logvar) = vae.forward(&x);

            let recon_loss = bce_reconstruction_loss(&recon, &x);
            let kl_loss = kl_divergence(&mu, &logvar);

            // Total loss = reconstruction + KL
            let loss = recon_loss.add(&kl_loss).unwrap();
            loss.backward();
            optimizer.step();

            total_recon_loss += recon_loss.tensor().item().unwrap();
            total_kl_loss += kl_loss.tensor().item().unwrap();
        }

        let avg_recon = total_recon_loss / batches_per_epoch as f64;
        let avg_kl = total_kl_loss / batches_per_epoch as f64;
        let avg_total = avg_recon + avg_kl;

        println!(
            "Epoch [{:2}/{}]  Total Loss: {:.4}  Recon Loss: {:.4}  KL Divergence: {:.4}",
            epoch, num_epochs, avg_total, avg_recon, avg_kl
        );
    }

    println!();
    println!("Training complete.");
}
