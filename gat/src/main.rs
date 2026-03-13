//! Graph Attention Network (GAT) Example
//!
//! Implements a GAT with multi-head attention for node classification on a
//! synthetic graph.
//! Reference: Velickovic et al., "Graph Attention Networks" (ICLR 2018).

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, CrossEntropyLoss, Module};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// GAT Layer
//
// For each edge (i, j), attention coefficient:
//   e_ij = LeakyReLU( a^T [W h_i || W h_j] )
//   alpha_ij = softmax_j(e_ij)
//   h'_i = sigma( sum_j alpha_ij * W h_j )
//
// Multi-head: K independent heads, outputs concatenated (or averaged).
// ---------------------------------------------------------------------------
struct GATLayer {
    /// Linear projection W per head: [in_features -> out_features]
    linears: Vec<Linear>,
    /// Attention vectors a per head: [2 * out_features -> 1]
    attn_linears: Vec<Linear>,
    num_heads: usize,
    out_features: usize,
    concat: bool, // true = concat heads, false = average
}

impl GATLayer {
    fn new(in_features: usize, out_features: usize, num_heads: usize, concat: bool) -> Self {
        let mut linears = Vec::new();
        let mut attn_linears = Vec::new();
        for _ in 0..num_heads {
            linears.push(Linear::no_bias(in_features, out_features));
            attn_linears.push(Linear::no_bias(2 * out_features, 1));
        }
        Self {
            linears,
            attn_linears,
            num_heads,
            out_features,
            concat,
        }
    }

    /// Forward pass.
    /// `adj`: [N, N] binary adjacency (1 = edge, 0 = no edge). Should include self-loops.
    /// `features`: [N, F_in] node features.
    /// Returns: [N, num_heads * out_features] if concat, else [N, out_features].
    fn forward_graph(&self, adj: &Variable, features: &Variable) -> Variable {
        let n = features.tensor().shape()[0];
        let adj_data = adj.tensor().to_vec_f64().unwrap();

        let mut head_outputs: Vec<Vec<f64>> = Vec::new();

        for head in 0..self.num_heads {
            // Project: Wh = features @ W^T   [N, out_features]
            let wh = self.linears[head].forward(features);
            let wh_data = wh.tensor().to_vec_f64().unwrap();
            let f_out = self.out_features;

            // Compute attention for all node pairs (only where adj > 0)
            // For each pair (i,j): concat [Wh_i || Wh_j] -> attn linear -> LeakyReLU
            let mut attn_scores = vec![f64::NEG_INFINITY; n * n];

            for i in 0..n {
                for j in 0..n {
                    if adj_data[i * n + j] > 0.5 {
                        // Concatenate Wh_i and Wh_j
                        let mut concat_vec = vec![0.0f64; 2 * f_out];
                        concat_vec[..f_out]
                            .copy_from_slice(&wh_data[i * f_out..(i + 1) * f_out]);
                        concat_vec[f_out..2 * f_out]
                            .copy_from_slice(&wh_data[j * f_out..(j + 1) * f_out]);

                        let cat_var =
                            Variable::new(Tensor::from_slice(&concat_vec, &[1, 2 * f_out]));
                        let e = self.attn_linears[head].forward(&cat_var);
                        let e_val = e.tensor().item().unwrap();
                        // LeakyReLU (negative slope = 0.2)
                        let e_val = if e_val > 0.0 { e_val } else { 0.2 * e_val };
                        attn_scores[i * n + j] = e_val;
                    }
                }
            }

            // Softmax over neighbours for each node
            let mut alpha = vec![0.0f64; n * n];
            for i in 0..n {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..n {
                    if attn_scores[i * n + j] > max_val {
                        max_val = attn_scores[i * n + j];
                    }
                }
                // If a node has no neighbours, skip
                if max_val == f64::NEG_INFINITY {
                    continue;
                }
                let mut sum_exp = 0.0f64;
                for j in 0..n {
                    if attn_scores[i * n + j] > f64::NEG_INFINITY {
                        let e = (attn_scores[i * n + j] - max_val).exp();
                        alpha[i * n + j] = e;
                        sum_exp += e;
                    }
                }
                if sum_exp > 0.0 {
                    for j in 0..n {
                        alpha[i * n + j] /= sum_exp;
                    }
                }
            }

            // Aggregate: h'_i = sum_j alpha_ij * Wh_j
            let mut head_out = vec![0.0f64; n * f_out];
            for i in 0..n {
                for j in 0..n {
                    if alpha[i * n + j] > 0.0 {
                        let a = alpha[i * n + j];
                        for f in 0..f_out {
                            head_out[i * f_out + f] += a * wh_data[j * f_out + f];
                        }
                    }
                }
            }

            head_outputs.push(head_out);
        }

        // Combine heads
        if self.concat {
            // Concatenate along feature dim: [N, num_heads * out_features]
            let total_f = self.num_heads * self.out_features;
            let mut combined = vec![0.0f64; n * total_f];
            for (h, head_out) in head_outputs.iter().enumerate() {
                for i in 0..n {
                    for f in 0..self.out_features {
                        combined[i * total_f + h * self.out_features + f] =
                            head_out[i * self.out_features + f];
                    }
                }
            }
            Variable::new(Tensor::from_slice(&combined, &[n, total_f]))
        } else {
            // Average heads: [N, out_features]
            let mut averaged = vec![0.0f64; n * self.out_features];
            for head_out in &head_outputs {
                for i in 0..n * self.out_features {
                    averaged[i] += head_out[i] / self.num_heads as f64;
                }
            }
            Variable::new(Tensor::from_slice(&averaged, &[n, self.out_features]))
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        for l in &self.linears {
            params.extend(l.parameters());
        }
        for l in &self.attn_linears {
            params.extend(l.parameters());
        }
        params
    }
}

// ---------------------------------------------------------------------------
// GAT Model
// ---------------------------------------------------------------------------
struct GAT {
    layer1: GATLayer,
    layer2: GATLayer,
}

impl GAT {
    fn new(in_features: usize, hidden: usize, num_classes: usize, heads: usize) -> Self {
        // Layer 1: multi-head with concatenation -> output is heads * hidden
        let layer1 = GATLayer::new(in_features, hidden, heads, true);
        // Layer 2: single head with averaging -> output is num_classes
        let layer2 = GATLayer::new(heads * hidden, num_classes, 1, false);
        Self { layer1, layer2 }
    }

    fn forward(&self, adj: &Variable, features: &Variable) -> Variable {
        // Layer 1 + ELU-like activation (using ReLU for simplicity)
        let h = self.layer1.forward_graph(adj, features);
        // Apply ReLU manually
        let h_data = h.tensor().to_vec_f64().unwrap();
        let relu_data: Vec<f64> = h_data.iter().map(|&x| x.max(0.0)).collect();
        let h = Variable::new(Tensor::from_slice(&relu_data, h.tensor().shape()));
        // Layer 2
        self.layer2.forward_graph(adj, &h)
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
fn random_adjacency_with_self_loops(num_nodes: usize, edge_prob: f64) -> Tensor {
    let mut rng = rand::thread_rng();
    let mut adj = vec![0.0f64; num_nodes * num_nodes];
    for i in 0..num_nodes {
        adj[i * num_nodes + i] = 1.0; // self-loop
        for j in (i + 1)..num_nodes {
            if rng.gen::<f64>() < edge_prob {
                adj[i * num_nodes + j] = 1.0;
                adj[j * num_nodes + i] = 1.0;
            }
        }
    }
    Tensor::from_slice(&adj, &[num_nodes, num_nodes])
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    println!("=== Graph Attention Network (GAT) — Node Classification ===\n");

    let num_nodes = 50;
    let num_features = 16;
    let hidden_dim = 8;
    let num_heads = 8;
    let num_classes = 4;
    let num_epochs = 30;
    let lr = 0.005;
    let edge_prob = 0.15;

    let adj = random_adjacency_with_self_loops(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj);
    let feat_var = Variable::new(features);
    let label_var = Variable::new(labels);

    let model = GAT::new(num_features, hidden_dim, num_classes, num_heads);
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), lr);

    let total_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!(
        "Graph: {} nodes, {} features, {} classes, {} attention heads",
        num_nodes, num_features, num_classes, num_heads
    );
    println!(
        "Model: GATLayer({}, {}, heads={}) -> ReLU -> GATLayer({}, {}, heads=1)",
        num_features, hidden_dim, num_heads, num_heads * hidden_dim, num_classes
    );
    println!("Total parameters: {}\n", total_params);

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        let logits = model.forward(&adj_var, &feat_var);
        let loss = criterion.forward(&logits, &label_var);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "Epoch [{:>3}/{}]  Loss: {:.4}",
                epoch + 1,
                num_epochs,
                loss_val
            );
        }
    }

    println!("\nTraining complete.");
}
