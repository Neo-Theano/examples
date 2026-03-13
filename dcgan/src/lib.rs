//! Deep Convolutional GAN (DCGAN) model definitions.
//!
//! - Generator: Linear(100, 256) -> ReLU -> ... -> Linear(1024, 784) -> Tanh
//! - Discriminator: Linear(784, 512) -> ReLU -> ... -> Linear(256, 1) -> Sigmoid

use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

pub struct Generator {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
    pub fc4: Linear,
}

impl Generator {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(100, 256),
            fc2: Linear::new(256, 512),
            fc3: Linear::new(512, 1024),
            fc4: Linear::new(1024, 784),
        }
    }

    pub fn forward(&self, z: &Variable) -> Variable {
        let h1 = self.fc1.forward(z).relu().unwrap();
        let h2 = self.fc2.forward(&h1).relu().unwrap();
        let h3 = self.fc3.forward(&h2).relu().unwrap();
        self.fc4.forward(&h3).tanh().unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params.extend(self.fc4.parameters());
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
        for (name, param) in self.fc3.named_parameters() {
            sd.insert(format!("{prefix}fc3.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc4.named_parameters() {
            sd.insert(format!("{prefix}fc4.{name}"), param.tensor().clone());
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
            fc3: Linear::from_tensors(
                sd[&format!("{prefix}fc3.weight")].clone(),
                Some(sd[&format!("{prefix}fc3.bias")].clone()),
            ),
            fc4: Linear::from_tensors(
                sd[&format!("{prefix}fc4.weight")].clone(),
                Some(sd[&format!("{prefix}fc4.bias")].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Discriminator
// ---------------------------------------------------------------------------

pub struct Discriminator {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

impl Discriminator {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(784, 512),
            fc2: Linear::new(512, 256),
            fc3: Linear::new(256, 1),
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        let h1 = self.fc1.forward(x).relu().unwrap();
        let h2 = self.fc2.forward(&h1).relu().unwrap();
        self.fc3.forward(&h2).sigmoid().unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
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
        for (name, param) in self.fc3.named_parameters() {
            sd.insert(format!("{prefix}fc3.{name}"), param.tensor().clone());
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
            fc3: Linear::from_tensors(
                sd[&format!("{prefix}fc3.weight")].clone(),
                Some(sd[&format!("{prefix}fc3.bias")].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// DCGAN (combined model)
// ---------------------------------------------------------------------------

pub struct DCGAN {
    pub generator: Generator,
    pub discriminator: Discriminator,
}

impl DCGAN {
    pub fn new() -> Self {
        Self {
            generator: Generator::new(),
            discriminator: Discriminator::new(),
        }
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = self.generator.state_dict("gen.");
        sd.extend(self.discriminator.state_dict("disc."));
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            generator: Generator::from_state_dict(sd, "gen."),
            discriminator: Discriminator::from_state_dict(sd, "disc."),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate random noise z ~ N(0, 1) of shape [batch_size, latent_dim].
pub fn random_noise(batch_size: usize, latent_dim: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let data: Vec<f64> = (0..batch_size * latent_dim)
        .map(|_| dist.sample(&mut rng))
        .collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, latent_dim]))
}

/// Generate synthetic "real" data — random values in [-1, 1] to match Tanh output range.
pub fn synthetic_real_data(batch_size: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..batch_size * 784)
        .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
        .collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 784]))
}

/// Create a target tensor filled with the given value.
pub fn target_tensor(batch_size: usize, value: f64) -> Variable {
    Variable::new(Tensor::full(&[batch_size, 1], value))
}
