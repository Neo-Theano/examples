//! Graph Convolutional Network (GCN) Example
//!
//! Implements a simple 2-layer GCN for node classification on a synthetic graph.
//! Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
//! Convolutional Networks" (ICLR 2017).

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, ReLU, CrossEntropyLoss, Module};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// GCN Layer: H' = sigma(A_hat @ H @ W)
// A_hat = D^{-1/2} A_tilde D^{-1/2}  where A_tilde = A + I
// ---------------------------------------------------------------------------
struct GCNLayer {
    weight: Linear,
}

impl GCNLayer {
    fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: Linear::no_bias(in_features, out_features),
        }
    }

    /// Forward pass: apply graph convolution.
    /// `adj_hat`: [N, N] normalised adjacency matrix (precomputed).
    /// `features`: [N, F_in] node feature matrix.
    /// Returns: [N, F_out]
    fn forward_graph(&self, adj_hat: &Variable, features: &Variable) -> Variable {
        // H_new = A_hat @ features @ W^T
        let support = adj_hat.matmul(features).unwrap(); // [N, F_in]
        self.weight.forward(&support) // [N, F_out]
    }

    fn parameters(&self) -> Vec<Variable> {
        self.weight.parameters()
    }
}

// ---------------------------------------------------------------------------
// GCN Model: 2-layer GCN
// ---------------------------------------------------------------------------
struct GCN {
    layer1: GCNLayer,
    layer2: GCNLayer,
    relu: ReLU,
}

impl GCN {
    fn new(in_features: usize, hidden: usize, num_classes: usize) -> Self {
        Self {
            layer1: GCNLayer::new(in_features, hidden),
            layer2: GCNLayer::new(hidden, num_classes),
            relu: ReLU,
        }
    }

    fn forward(&self, adj_hat: &Variable, features: &Variable) -> Variable {
        let h = self.layer1.forward_graph(adj_hat, features);
        let h = self.relu.forward(&h);
        self.layer2.forward_graph(adj_hat, &h)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.layer1.parameters();
        p.extend(self.layer2.parameters());
        p
    }
}

// ---------------------------------------------------------------------------
// Synthetic graph helpers
// ---------------------------------------------------------------------------

/// Build a random adjacency matrix with self-loops and symmetry, then compute
/// the normalized adjacency A_hat = D^{-1/2} A_tilde D^{-1/2}.
fn random_normalized_adjacency(num_nodes: usize, edge_prob: f64) -> Tensor {
    let mut rng = rand::thread_rng();
    let mut adj = vec![0.0f64; num_nodes * num_nodes];

    // Random symmetric edges + self-loops
    for i in 0..num_nodes {
        adj[i * num_nodes + i] = 1.0; // self-loop
        for j in (i + 1)..num_nodes {
            if rng.gen::<f64>() < edge_prob {
                adj[i * num_nodes + j] = 1.0;
                adj[j * num_nodes + i] = 1.0;
            }
        }
    }

    // Compute degree vector
    let mut deg = vec![0.0f64; num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            deg[i] += adj[i * num_nodes + j];
        }
    }

    // D^{-1/2} A D^{-1/2}
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

fn random_features(num_nodes: usize, num_features: usize) -> Tensor {
    let numel = num_nodes * num_features;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Tensor::from_slice(&data, &[num_nodes, num_features])
}

fn random_labels(num_nodes: usize, num_classes: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..num_nodes)
        .map(|_| rng.gen_range(0..num_classes) as f64)
        .collect();
    Tensor::from_slice(&data, &[num_nodes])
}

fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    println!("=== Graph Convolutional Network (GCN) — Node Classification ===\n");

    let num_nodes = 100;
    let num_features = 16;
    let hidden_dim = 16;
    let num_classes = 5;
    let num_epochs = 50;
    let lr = 0.01;
    let edge_prob = 0.1;

    // Generate synthetic graph
    let adj = random_normalized_adjacency(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj);
    let feat_var = Variable::new(features);
    let label_var = Variable::new(labels.clone());

    let model = GCN::new(num_features, hidden_dim, num_classes);
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), lr);

    println!(
        "Graph: {} nodes, {} features, {} classes",
        num_nodes, num_features, num_classes
    );
    println!(
        "Model: GCNLayer({}, {}) -> ReLU -> GCNLayer({}, {})\n",
        num_features, hidden_dim, hidden_dim, num_classes
    );

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        let logits = model.forward(&adj_var, &feat_var);
        let loss = criterion.forward(&logits, &label_var);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        let accuracy = compute_accuracy(logits.tensor(), &labels);

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Epoch [{:>3}/{}]  Loss: {:.4}  Accuracy: {:.2}%",
                epoch + 1,
                num_epochs,
                loss_val,
                accuracy * 100.0
            );
        }
    }

    println!("\nTraining complete.");
}
