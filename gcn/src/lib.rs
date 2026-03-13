//! Graph Convolutional Network (GCN) model definitions.
//!
//! Implements a 2-layer GCN for node classification.
//! GCNLayer: H' = sigma(A_hat @ H @ W)
//! Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
//! Convolutional Networks" (ICLR 2017).

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, ReLU, Module};

// ---------------------------------------------------------------------------
// GCN Layer
// ---------------------------------------------------------------------------

pub struct GCNLayer {
    pub weight: Linear,
}

impl GCNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: Linear::no_bias(in_features, out_features),
        }
    }

    /// Forward pass: apply graph convolution.
    /// `adj_hat`: [N, N] normalised adjacency matrix.
    /// `features`: [N, F_in] node feature matrix.
    /// Returns: [N, F_out]
    pub fn forward_graph(&self, adj_hat: &Variable, features: &Variable) -> Variable {
        let support = adj_hat.matmul(features).unwrap();
        self.weight.forward(&support)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        self.weight.parameters()
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.weight.named_parameters() {
            sd.insert(format!("{prefix}{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        Self {
            weight: Linear::from_tensors(
                sd[&format!("{prefix}weight")].clone(),
                None,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// GCN Model
// ---------------------------------------------------------------------------

pub struct GCN {
    pub layer1: GCNLayer,
    pub layer2: GCNLayer,
    pub relu: ReLU,
}

impl GCN {
    pub fn new(in_features: usize, hidden: usize, num_classes: usize) -> Self {
        Self {
            layer1: GCNLayer::new(in_features, hidden),
            layer2: GCNLayer::new(hidden, num_classes),
            relu: ReLU,
        }
    }

    pub fn forward(&self, adj_hat: &Variable, features: &Variable) -> Variable {
        let h = self.layer1.forward_graph(adj_hat, features);
        let h = self.relu.forward(&h);
        self.layer2.forward_graph(adj_hat, &h)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut p = self.layer1.parameters();
        p.extend(self.layer2.parameters());
        p
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = self.layer1.state_dict("layer1.");
        sd.extend(self.layer2.state_dict("layer2."));
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            layer1: GCNLayer::from_state_dict(sd, "layer1."),
            layer2: GCNLayer::from_state_dict(sd, "layer2."),
            relu: ReLU,
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic graph helpers
// ---------------------------------------------------------------------------

/// Build a random adjacency matrix with self-loops and symmetry, then compute
/// the normalized adjacency A_hat = D^{-1/2} A_tilde D^{-1/2}.
pub fn random_normalized_adjacency(num_nodes: usize, edge_prob: f64) -> Tensor {
    let mut rng = rand::thread_rng();
    let mut adj = vec![0.0f64; num_nodes * num_nodes];

    for i in 0..num_nodes {
        adj[i * num_nodes + i] = 1.0;
        for j in (i + 1)..num_nodes {
            if rng.gen::<f64>() < edge_prob {
                adj[i * num_nodes + j] = 1.0;
                adj[j * num_nodes + i] = 1.0;
            }
        }
    }

    let mut deg = vec![0.0f64; num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            deg[i] += adj[i * num_nodes + j];
        }
    }

    let mut norm_adj = vec![0.0f64; num_nodes * num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if adj[i * num_nodes + j] > 0.0 {
                norm_adj[i * num_nodes + j] =
                    adj[i * num_nodes + j] / (deg[i].sqrt() * deg[j].sqrt());
            }
        }
    }

    Tensor::from_slice(&norm_adj, &[num_nodes, num_nodes])
}

pub fn random_features(num_nodes: usize, num_features: usize) -> Tensor {
    let numel = num_nodes * num_features;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Tensor::from_slice(&data, &[num_nodes, num_features])
}

pub fn random_labels(num_nodes: usize, num_classes: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..num_nodes)
        .map(|_| rng.gen_range(0..num_classes) as f64)
        .collect();
    Tensor::from_slice(&data, &[num_nodes])
}

pub fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
    let logits_data = logits.to_vec_f64().unwrap();
    let targets_data = targets.to_vec_f64().unwrap();
    let num_nodes = targets.shape()[0];
    let num_classes = logits.shape()[1];

    let mut correct = 0;
    for i in 0..num_nodes {
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;
        for c in 0..num_classes {
            let val = logits_data[i * num_classes + c];
            if val > max_val {
                max_val = val;
                max_idx = c;
            }
        }
        if max_idx == targets_data[i] as usize {
            correct += 1;
        }
    }
    correct as f64 / num_nodes as f64
}
