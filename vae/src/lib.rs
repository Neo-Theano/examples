//! Variational Autoencoder (VAE) model definitions.
//!
//! Implements a VAE with:
//! - Encoder: Linear(784, 400) -> ReLU -> (mu, logvar) heads
//! - Reparameterization trick: z = mu + std * eps
//! - Decoder: Linear(20, 400) -> ReLU -> Linear(400, 784) -> Sigmoid
//! - Loss: BCE reconstruction + KL divergence

use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Encoder {
    fc1: Linear,
    fc_mu: Linear,
    fc_logvar: Linear,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(784, 400),
            fc_mu: Linear::new(400, 20),
            fc_logvar: Linear::new(400, 20),
        }
    }

    pub fn forward(&self, x: &Variable) -> (Variable, Variable) {
        let h = self.fc1.forward(x).relu().unwrap();
        let mu = self.fc_mu.forward(&h);
        let logvar = self.fc_logvar.forward(&h);
        (mu, logvar)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc_mu.parameters());
        params.extend(self.fc_logvar.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.fc1.named_parameters() {
            sd.insert(format!("{prefix}fc1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc_mu.named_parameters() {
            sd.insert(format!("{prefix}fc_mu.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc_logvar.named_parameters() {
            sd.insert(format!("{prefix}fc_logvar.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        Self {
            fc1: Linear::from_tensors(
                sd[&format!("{prefix}fc1.weight")].clone(),
                Some(sd[&format!("{prefix}fc1.bias")].clone()),
            ),
            fc_mu: Linear::from_tensors(
                sd[&format!("{prefix}fc_mu.weight")].clone(),
                Some(sd[&format!("{prefix}fc_mu.bias")].clone()),
            ),
            fc_logvar: Linear::from_tensors(
                sd[&format!("{prefix}fc_logvar.weight")].clone(),
                Some(sd[&format!("{prefix}fc_logvar.bias")].clone()),
            ),
        }
    }
}

pub struct Decoder {
    fc1: Linear,
    fc2: Linear,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(20, 400),
            fc2: Linear::new(400, 784),
        }
    }

    pub fn forward(&self, z: &Variable) -> Variable {
        let h = self.fc1.forward(z).relu().unwrap();
        self.fc2.forward(&h).sigmoid().unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.fc1.named_parameters() {
            sd.insert(format!("{prefix}fc1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc2.named_parameters() {
            sd.insert(format!("{prefix}fc2.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        Self {
            fc1: Linear::from_tensors(
                sd[&format!("{prefix}fc1.weight")].clone(),
                Some(sd[&format!("{prefix}fc1.bias")].clone()),
            ),
            fc2: Linear::from_tensors(
                sd[&format!("{prefix}fc2.weight")].clone(),
                Some(sd[&format!("{prefix}fc2.bias")].clone()),
            ),
        }
    }
}

pub struct VAE {
    pub encoder: Encoder,
    pub decoder: Decoder,
}

impl VAE {
    pub fn new() -> Self {
        Self {
            encoder: Encoder::new(),
            decoder: Decoder::new(),
        }
    }

    /// Forward pass with reparameterization trick.
    /// Returns (reconstruction, mu, logvar).
    pub fn forward(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let (mu, logvar) = self.encoder.forward(x);

        // Reparameterization trick: z = mu + std * eps
        let std = logvar.mul_scalar(0.5).unwrap().exp().unwrap();
        let eps = random_normal_like(&std);
        let z = mu.add(&std.mul(&eps).unwrap()).unwrap();

        let recon = self.decoder.forward(&z);
        (recon, mu, logvar)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = self.encoder.state_dict("encoder.");
        sd.extend(self.decoder.state_dict("decoder."));
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            encoder: Encoder::from_state_dict(sd, "encoder."),
            decoder: Decoder::from_state_dict(sd, "decoder."),
        }
    }
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

/// BCE reconstruction loss (computed element-wise for numerical stability).
pub fn bce_reconstruction_loss(recon: &Variable, target: &Variable) -> Variable {
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
pub fn kl_divergence(mu: &Variable, logvar: &Variable) -> Variable {
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
pub fn random_normal_like(v: &Variable) -> Variable {
    let shape = v.tensor().shape();
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let data: Vec<f64> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Variable::new(Tensor::from_slice(&data, shape))
}

/// Generate a batch of synthetic flattened 28x28 images with pixel values in [0, 1].
pub fn synthetic_batch(batch_size: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let numel = batch_size * 784;
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 784]))
}
