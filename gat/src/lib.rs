//! Graph Attention Network (GAT) model definitions.
//!
//! Implements a GAT with multi-head attention for node classification.
//! Reference: Velickovic et al., "Graph Attention Networks" (ICLR 2018).

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// ---------------------------------------------------------------------------
// GAT Layer
// ---------------------------------------------------------------------------

pub struct GATLayer {
    /// Linear projection W per head: [in_features -> out_features]
    pub linears: Vec<Linear>,
    /// Attention vectors a per head: [2 * out_features -> 1]
    pub attn_linears: Vec<Linear>,
    pub num_heads: usize,
    pub out_features: usize,
    pub concat: bool,
}

impl GATLayer {
    pub fn new(in_features: usize, out_features: usize, num_heads: usize, concat: bool) -> Self {
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
    pub fn forward_graph(&self, adj: &Variable, features: &Variable) -> Variable {
        let n = features.tensor().shape()[0];
        let adj_data = adj.tensor().to_vec_f64().unwrap();

        let mut head_outputs: Vec<Vec<f64>> = Vec::new();

        for head in 0..self.num_heads {
            let wh = self.linears[head].forward(features);
            let wh_data = wh.tensor().to_vec_f64().unwrap();
            let f_out = self.out_features;

            let mut attn_scores = vec![f64::NEG_INFINITY; n * n];

            for i in 0..n {
                for j in 0..n {
                    if adj_data[i * n + j] > 0.5 {
                        let mut concat_vec = vec![0.0f64; 2 * f_out];
                        concat_vec[..f_out]
                            .copy_from_slice(&wh_data[i * f_out..(i + 1) * f_out]);
                        concat_vec[f_out..2 * f_out]
                            .copy_from_slice(&wh_data[j * f_out..(j + 1) * f_out]);

                        let cat_var =
                            Variable::new(Tensor::from_slice(&concat_vec, &[1, 2 * f_out]));
                        let e = self.attn_linears[head].forward(&cat_var);
                        let e_val = e.tensor().item().unwrap();
                        let e_val = if e_val > 0.0 { e_val } else { 0.2 * e_val };
                        attn_scores[i * n + j] = e_val;
                    }
                }
            }

            let mut alpha = vec![0.0f64; n * n];
            for i in 0..n {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..n {
                    if attn_scores[i * n + j] > max_val {
                        max_val = attn_scores[i * n + j];
                    }
                }
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

        if self.concat {
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
            let mut averaged = vec![0.0f64; n * self.out_features];
            for head_out in &head_outputs {
                for i in 0..n * self.out_features {
                    averaged[i] += head_out[i] / self.num_heads as f64;
                }
            }
            Variable::new(Tensor::from_slice(&averaged, &[n, self.out_features]))
        }
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        for l in &self.linears {
            params.extend(l.parameters());
        }
        for l in &self.attn_linears {
            params.extend(l.parameters());
        }
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // Store hyperparameters
        sd.insert(
            format!("{prefix}num_heads"),
            Tensor::from_slice(&[self.num_heads as f64], &[1]),
        );
        sd.insert(
            format!("{prefix}out_features"),
            Tensor::from_slice(&[self.out_features as f64], &[1]),
        );
        sd.insert(
            format!("{prefix}concat"),
            Tensor::from_slice(&[if self.concat { 1.0 } else { 0.0 }], &[1]),
        );

        // Per-head W projection weights
        for (h, linear) in self.linears.iter().enumerate() {
            for (name, param) in linear.named_parameters() {
                sd.insert(
                    format!("{prefix}head{h}.W.{name}"),
                    param.tensor().clone(),
                );
            }
        }

        // Per-head attention weights
        for (h, attn_linear) in self.attn_linears.iter().enumerate() {
            for (name, param) in attn_linear.named_parameters() {
                sd.insert(
                    format!("{prefix}head{h}.attn.{name}"),
                    param.tensor().clone(),
                );
            }
        }

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        let num_heads = sd[&format!("{prefix}num_heads")].to_vec_f64().unwrap()[0] as usize;
        let out_features = sd[&format!("{prefix}out_features")].to_vec_f64().unwrap()[0] as usize;
        let concat = sd[&format!("{prefix}concat")].to_vec_f64().unwrap()[0] > 0.5;

        let mut linears = Vec::new();
        let mut attn_linears = Vec::new();

        for h in 0..num_heads {
            linears.push(Linear::from_tensors(
                sd[&format!("{prefix}head{h}.W.weight")].clone(),
                None,
            ));
            attn_linears.push(Linear::from_tensors(
                sd[&format!("{prefix}head{h}.attn.weight")].clone(),
                None,
            ));
        }

        Self {
            linears,
            attn_linears,
            num_heads,
            out_features,
            concat,
        }
    }
}

// ---------------------------------------------------------------------------
// GAT Model
// ---------------------------------------------------------------------------

pub struct GAT {
    pub layer1: GATLayer,
    pub layer2: GATLayer,
}

impl GAT {
    pub fn new(in_features: usize, hidden: usize, num_classes: usize, heads: usize) -> Self {
        let layer1 = GATLayer::new(in_features, hidden, heads, true);
        let layer2 = GATLayer::new(heads * hidden, num_classes, 1, false);
        Self { layer1, layer2 }
    }

    pub fn forward(&self, adj: &Variable, features: &Variable) -> Variable {
        let h = self.layer1.forward_graph(adj, features);
        let h_data = h.tensor().to_vec_f64().unwrap();
        let relu_data: Vec<f64> = h_data.iter().map(|&x| x.max(0.0)).collect();
        let h = Variable::new(Tensor::from_slice(&relu_data, h.tensor().shape()));
        self.layer2.forward_graph(adj, &h)
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
            layer1: GATLayer::from_state_dict(sd, "layer1."),
            layer2: GATLayer::from_state_dict(sd, "layer2."),
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic graph helpers
// ---------------------------------------------------------------------------

pub fn random_adjacency_with_self_loops(num_nodes: usize, edge_prob: f64) -> Tensor {
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
    Tensor::from_slice(&adj, &[num_nodes, num_nodes])
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
