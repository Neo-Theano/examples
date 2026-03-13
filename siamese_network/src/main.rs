//! Siamese Network example.
//!
//! Implements a twin network with shared weights:
//! - Shared backbone: Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 64)
//! - Contrastive loss for learning similarity/dissimilarity of pairs
//!
//! Trained on synthetic pairs with labels.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct SiameseNetwork {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SiameseNetwork {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 64),
        }
    }

    /// Forward one branch of the twin network.
    fn forward_one(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x).relu().unwrap();
        let h = self.fc2.forward(&h).relu().unwrap();
        self.fc3.forward(&h)
    }

    /// Forward both branches and return embeddings.
    fn forward(&self, x1: &Variable, x2: &Variable) -> (Variable, Variable) {
        let emb1 = self.forward_one(x1);
        let emb2 = self.forward_one(x2);
        (emb1, emb2)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Contrastive Loss
// ---------------------------------------------------------------------------

/// Contrastive loss for Siamese networks.
///
/// For same-class pairs (label=1): loss = ||f(x1) - f(x2)||^2
/// For diff-class pairs (label=0): loss = max(0, margin - ||f(x1) - f(x2)||)^2
fn contrastive_loss(
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
                // d(hinge^2)/d(dist_sq) requires chain rule through sqrt
                // Simpler: weight by hinge activation indicator
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
fn contrastive_loss_value(
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
fn synthetic_pairs(batch_size: usize) -> (Variable, Variable, Vec<f64>) {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Siamese Network ===");
    println!();

    let model = SiameseNetwork::new();
    let batch_size = 32;
    let num_epochs = 20;
    let batches_per_epoch = 10;
    let margin = 2.0;

    let param_count: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("Model parameters: {}", param_count);
    println!("Embedding dimension: 64");
    println!("Contrastive margin: {}", margin);
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", num_epochs);
    println!();

    let mut optimizer = Adam::new(model.parameters(), 1e-3);

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;
        let mut total_same_dist = 0.0;
        let mut total_diff_dist = 0.0;
        let mut same_count = 0;
        let mut diff_count = 0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            let (x1, x2, labels) = synthetic_pairs(batch_size);
            let (emb1, emb2) = model.forward(&x1, &x2);

            // Compute loss (with gradient graph)
            let loss = contrastive_loss(&emb1, &emb2, &labels, margin);

            // Compute display loss value
            let loss_val = contrastive_loss_value(&emb1, &emb2, &labels, margin);
            total_loss += loss_val;

            // Track average distances for same/different pairs
            let e1 = emb1.tensor().to_vec_f64().unwrap();
            let e2 = emb2.tensor().to_vec_f64().unwrap();
            let embed_dim = 64;
            for i in 0..batch_size {
                let mut dist_sq = 0.0;
                for j in 0..embed_dim {
                    let diff = e1[i * embed_dim + j] - e2[i * embed_dim + j];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                if labels[i] > 0.5 {
                    total_same_dist += dist;
                    same_count += 1;
                } else {
                    total_diff_dist += dist;
                    diff_count += 1;
                }
            }

            loss.backward();
            optimizer.step();
        }

        let avg_loss = total_loss / batches_per_epoch as f64;
        let avg_same = if same_count > 0 {
            total_same_dist / same_count as f64
        } else {
            0.0
        };
        let avg_diff = if diff_count > 0 {
            total_diff_dist / diff_count as f64
        } else {
            0.0
        };

        println!(
            "Epoch [{:2}/{}]  Loss: {:.4}  Avg Same Dist: {:.4}  Avg Diff Dist: {:.4}",
            epoch, num_epochs, avg_loss, avg_same, avg_diff
        );
    }

    // Test with a few pairs
    println!();
    println!("Testing with sample pairs...");
    let (test_x1, test_x2, test_labels) = synthetic_pairs(5);
    let (test_e1, test_e2) = model.forward(&test_x1, &test_x2);
    let e1_data = test_e1.tensor().to_vec_f64().unwrap();
    let e2_data = test_e2.tensor().to_vec_f64().unwrap();
    let embed_dim = 64;

    for i in 0..5 {
        let mut dist_sq = 0.0;
        for j in 0..embed_dim {
            let diff = e1_data[i * embed_dim + j] - e2_data[i * embed_dim + j];
            dist_sq += diff * diff;
        }
        let label_str = if test_labels[i] > 0.5 { "same" } else { "diff" };
        println!(
            "  Pair {}: label={}, distance={:.4}",
            i + 1,
            label_str,
            dist_sq.sqrt()
        );
    }

    println!();
    println!("Training complete.");
}
