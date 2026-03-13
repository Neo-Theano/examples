//! Siamese Network model definitions.
//!
//! Twin network with shared weights:
//! - Shared backbone: Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 64)
//! - Contrastive loss for learning similarity/dissimilarity of pairs

use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SiameseNetwork {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

impl SiameseNetwork {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 64),
        }
    }

    /// Forward one branch of the twin network.
    pub fn forward_one(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x).relu().unwrap();
        let h = self.fc2.forward(&h).relu().unwrap();
        self.fc3.forward(&h)
    }

    /// Forward both branches and return embeddings.
    pub fn forward(&self, x1: &Variable, x2: &Variable) -> (Variable, Variable) {
        let emb1 = self.forward_one(x1);
        let emb2 = self.forward_one(x2);
        (emb1, emb2)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.fc1.named_parameters() {
            sd.insert(format!("fc1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc2.named_parameters() {
            sd.insert(format!("fc2.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc3.named_parameters() {
            sd.insert(format!("fc3.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            fc1: Linear::from_tensors(
                sd["fc1.weight"].clone(),
                Some(sd["fc1.bias"].clone()),
            ),
            fc2: Linear::from_tensors(
                sd["fc2.weight"].clone(),
                Some(sd["fc2.bias"].clone()),
            ),
            fc3: Linear::from_tensors(
                sd["fc3.weight"].clone(),
                Some(sd["fc3.bias"].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Contrastive Loss
// ---------------------------------------------------------------------------

/// Contrastive loss for Siamese networks.
///
/// For same-class pairs (label=1): loss = ||f(x1) - f(x2)||^2
/// For diff-class pairs (label=0): loss = max(0, margin - ||f(x1) - f(x2)||)^2
pub fn contrastive_loss(
    emb1: &Variable,
    emb2: &Variable,
    labels: &[f64],
    margin: f64,
) -> Variable {
    let emb1_data = emb1.tensor().to_vec_f64().unwrap();
    let emb2_data = emb2.tensor().to_vec_f64().unwrap();

    let batch_size = emb1.tensor().shape()[0];
    let embed_dim = emb1.tensor().shape()[1];

    let mut total_loss = 0.0;

    for i in 0..batch_size {
        // Compute Euclidean distance for this pair
        let mut dist_sq = 0.0;
        for j in 0..embed_dim {
            let diff = emb1_data[i * embed_dim + j] - emb2_data[i * embed_dim + j];
            dist_sq += diff * diff;
        }
        let dist = dist_sq.sqrt();

        let label = labels[i];
        if label > 0.5 {
            // Same class: minimize distance
            total_loss += dist_sq;
        } else {
            // Different class: push apart beyond margin
            let hinge = (margin - dist).max(0.0);
            total_loss += hinge * hinge;
        }
    }

    // For gradient flow, we also build the Variable computation graph
    let diff = emb1.sub(emb2).unwrap();
    let diff_sq = diff.mul(&diff).unwrap();
    // Sum over embedding dimension to get per-pair distances
    let dist_sq_per_pair = diff_sq.sum_dim(1, true).unwrap();

    // Compute loss per pair using Variable ops
    let dist_sq_data = dist_sq_per_pair.tensor().to_vec_f64().unwrap();
    let mut loss_weights = vec![0.0f64; batch_size];

    for i in 0..batch_size {
        let d_sq = dist_sq_data[i];
        let d = d_sq.sqrt();
        if labels[i] > 0.5 {
            // Same class: gradient weight = 1.0
            loss_weights[i] = 1.0;
        } else {
            // Different class: gradient weight depends on hinge
            let hinge = (margin - d).max(0.0);
            if hinge > 0.0 {
                loss_weights[i] = 1.0;
            } else {
                loss_weights[i] = 0.0;
            }
        }
    }

    let weights_var = Variable::new(Tensor::from_slice(&loss_weights, &[batch_size, 1]));
    let weighted = dist_sq_per_pair.mul(&weights_var).unwrap();
    let batch_loss = weighted.sum().unwrap().mul_scalar(1.0 / batch_size as f64).unwrap();

    // Return the scalar loss (use the manually computed value for display)
    // but the Variable graph for backprop
    let _ = total_loss; // used only for display via separate path
    batch_loss
}

/// Compute contrastive loss value for display (no graph).
pub fn contrastive_loss_value(
    emb1: &Variable,
    emb2: &Variable,
    labels: &[f64],
    margin: f64,
) -> f64 {
    let emb1_data = emb1.tensor().to_vec_f64().unwrap();
    let emb2_data = emb2.tensor().to_vec_f64().unwrap();
    let batch_size = emb1.tensor().shape()[0];
    let embed_dim = emb1.tensor().shape()[1];

    let mut total_loss = 0.0;
    for i in 0..batch_size {
        let mut dist_sq = 0.0;
        for j in 0..embed_dim {
            let diff = emb1_data[i * embed_dim + j] - emb2_data[i * embed_dim + j];
            dist_sq += diff * diff;
        }
        let dist = dist_sq.sqrt();
        if labels[i] > 0.5 {
            total_loss += dist_sq;
        } else {
            let hinge = (margin - dist).max(0.0);
            total_loss += hinge * hinge;
        }
    }
    total_loss / batch_size as f64
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate synthetic pairs with labels.
/// Same-class pairs: both from the same random cluster.
/// Different-class pairs: from different random clusters.
pub fn synthetic_pairs(batch_size: usize) -> (Variable, Variable, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 0.3).unwrap();

    let mut x1_data = vec![0.0f64; batch_size * 784];
    let mut x2_data = vec![0.0f64; batch_size * 784];
    let mut labels = vec![0.0f64; batch_size];

    for i in 0..batch_size {
        let is_same = rng.gen_bool(0.5);
        labels[i] = if is_same { 1.0 } else { 0.0 };

        // Generate cluster center for x1
        let center1: Vec<f64> = (0..784).map(|_| rng.gen::<f64>()).collect();

        if is_same {
            // Same class: x2 is a noisy version of the same center
            for j in 0..784 {
                x1_data[i * 784 + j] = center1[j] + dist.sample(&mut rng);
                x2_data[i * 784 + j] = center1[j] + dist.sample(&mut rng);
            }
        } else {
            // Different class: x2 from a different random center
            let center2: Vec<f64> = (0..784).map(|_| rng.gen::<f64>()).collect();
            for j in 0..784 {
                x1_data[i * 784 + j] = center1[j] + dist.sample(&mut rng);
                x2_data[i * 784 + j] = center2[j] + dist.sample(&mut rng);
            }
        }
    }

    let x1 = Variable::new(Tensor::from_slice(&x1_data, &[batch_size, 784]));
    let x2 = Variable::new(Tensor::from_slice(&x2_data, &[batch_size, 784]));
    (x1, x2, labels)
}
