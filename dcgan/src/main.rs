//! Deep Convolutional GAN (DCGAN) example.
//!
//! Implements a GAN with linear layers simulating transposed convolutions:
//! - Generator: Linear(100, 256) -> ReLU -> ... -> Linear(1024, 784) -> Tanh
//! - Discriminator: Linear(784, 512) -> ReLU -> ... -> Linear(256, 1) -> Sigmoid
//!
//! Trained on synthetic flattened 28x28 images.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module, BCELoss};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

struct Generator {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    fc4: Linear,
}

impl Generator {
    fn new() -> Self {
        Self {
            fc1: Linear::new(100, 256),
            fc2: Linear::new(256, 512),
            fc3: Linear::new(512, 1024),
            fc4: Linear::new(1024, 784),
        }
    }

    fn forward(&self, z: &Variable) -> Variable {
        let h1 = self.fc1.forward(z).relu().unwrap();
        let h2 = self.fc2.forward(&h1).relu().unwrap();
        let h3 = self.fc3.forward(&h2).relu().unwrap();
        self.fc4.forward(&h3).tanh().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params.extend(self.fc4.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Discriminator
// ---------------------------------------------------------------------------

struct Discriminator {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl Discriminator {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 512),
            fc2: Linear::new(512, 256),
            fc3: Linear::new(256, 1),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h1 = self.fc1.forward(x).relu().unwrap();
        let h2 = self.fc2.forward(&h1).relu().unwrap();
        self.fc3.forward(&h2).sigmoid().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate random noise z ~ N(0, 1) of shape [batch_size, latent_dim].
fn random_noise(batch_size: usize, latent_dim: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let data: Vec<f64> = (0..batch_size * latent_dim)
        .map(|_| dist.sample(&mut rng))
        .collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, latent_dim]))
}

/// Generate synthetic "real" data — random values in [-1, 1] to match Tanh output range.
fn synthetic_real_data(batch_size: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..batch_size * 784)
        .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
        .collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 784]))
}

/// Create a target tensor filled with the given value.
fn target_tensor(batch_size: usize, value: f64) -> Variable {
    Variable::new(Tensor::full(&[batch_size, 1], value))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Deep Convolutional GAN (DCGAN) ===");
    println!();

    let generator = Generator::new();
    let discriminator = Discriminator::new();

    let g_params = generator.parameters();
    let d_params = discriminator.parameters();

    let g_param_count: usize = g_params.iter().map(|p| p.tensor().numel()).sum();
    let d_param_count: usize = d_params.iter().map(|p| p.tensor().numel()).sum();

    let mut g_optimizer = Adam::new(g_params, 2e-4).betas(0.5, 0.999);
    let mut d_optimizer = Adam::new(d_params, 2e-4).betas(0.5, 0.999);

    let bce = BCELoss::new();
    let batch_size = 64;
    let latent_dim = 100;
    let num_epochs = 20;
    let batches_per_epoch = 10;

    println!("Generator parameters: {}", g_param_count);
    println!("Discriminator parameters: {}", d_param_count);
    println!("Batch size: {}", batch_size);
    println!("Latent dimension: {}", latent_dim);
    println!("Epochs: {}", num_epochs);
    println!();

    for epoch in 1..=num_epochs {
        let mut total_d_loss = 0.0;
        let mut total_g_loss = 0.0;

        for _ in 0..batches_per_epoch {
            // ---------------------------------------------------------------
            // Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            // ---------------------------------------------------------------
            d_optimizer.zero_grad();

            // Real data
            let real_data = synthetic_real_data(batch_size);
            let real_labels = target_tensor(batch_size, 1.0);
            let d_real_output = discriminator.forward(&real_data);
            let d_real_loss = bce.forward(&d_real_output, &real_labels);

            // Fake data
            let noise = random_noise(batch_size, latent_dim);
            let fake_data = generator.forward(&noise);
            let fake_labels = target_tensor(batch_size, 0.0);
            let d_fake_output = discriminator.forward(&fake_data.detach());
            let d_fake_loss = bce.forward(&d_fake_output, &fake_labels);

            let d_loss_val = d_real_loss.tensor().item().unwrap()
                + d_fake_loss.tensor().item().unwrap();
            total_d_loss += d_loss_val;

            // Backward for discriminator (sum of both losses)
            let d_loss = d_real_loss.add(&d_fake_loss).unwrap();
            d_loss.backward();
            d_optimizer.step();

            // ---------------------------------------------------------------
            // Train Generator: maximize log(D(G(z)))
            // ---------------------------------------------------------------
            g_optimizer.zero_grad();

            let noise = random_noise(batch_size, latent_dim);
            let fake_data = generator.forward(&noise);
            let real_labels_for_g = target_tensor(batch_size, 1.0);
            let g_output = discriminator.forward(&fake_data);
            let g_loss = bce.forward(&g_output, &real_labels_for_g);

            let g_loss_val = g_loss.tensor().item().unwrap();
            total_g_loss += g_loss_val;

            g_loss.backward();
            g_optimizer.step();
        }

        let avg_d_loss = total_d_loss / batches_per_epoch as f64;
        let avg_g_loss = total_g_loss / batches_per_epoch as f64;

        println!(
            "Epoch [{:2}/{}]  D Loss: {:.4}  G Loss: {:.4}",
            epoch, num_epochs, avg_d_loss, avg_g_loss
        );
    }

    // Generate a sample
    println!();
    println!("Generating sample from trained generator...");
    let sample_noise = random_noise(1, latent_dim);
    let sample = generator.forward(&sample_noise);
    let sample_data = sample.tensor().to_vec_f64().unwrap();
    let sample_min = sample_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let sample_max = sample_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Sample stats: min={:.4}, max={:.4}, shape={:?}",
        sample_min,
        sample_max,
        sample.tensor().shape()
    );
    println!();
    println!("Training complete.");
}
