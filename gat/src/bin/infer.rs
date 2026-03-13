//! GAT inference example.
//!
//! Loads a trained GAT model, creates a synthetic graph, classifies nodes,
//! and prints attention weights.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::Module;
use theano_serialize::load_state_dict;

use gat::{GAT, random_adjacency_with_self_loops, random_features, random_labels};

fn main() {
    println!("=== GAT Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("gat_model.safetensors")
        .expect("Model file not found. Run training first: cargo run --bin gat");
    let sd = load_state_dict(&bytes).unwrap();
    let model = GAT::from_state_dict(&sd);
    println!("Model loaded from gat_model.safetensors");

    // Create a synthetic graph for inference
    let num_nodes = 10;
    let num_features = 16;
    let num_classes = 4;
    let edge_prob = 0.3;

    let adj = random_adjacency_with_self_loops(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj.clone());
    let feat_var = Variable::new(features);

    // Run inference
    println!("\n--- Node classification on synthetic graph ---");
    println!("Graph: {} nodes, {} features, {} classes", num_nodes, num_features, num_classes);

    let logits = model.forward(&adj_var, &feat_var);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Print predictions for each node
    println!("\n--- Node predictions ---");
    let labels_data = labels.to_vec_f64().unwrap();
    let mut correct = 0;
    for i in 0..num_nodes {
        let offset = i * num_classes;
        let mut best_class = 0;
        let mut best_val = f64::NEG_INFINITY;
        for c in 0..num_classes {
            if logits_data[offset + c] > best_val {
                best_val = logits_data[offset + c];
                best_class = c;
            }
        }
        if best_class == labels_data[i] as usize {
            correct += 1;
        }
        let class_logits: Vec<String> = (0..num_classes)
            .map(|c| format!("{:.3}", logits_data[offset + c]))
            .collect();
        println!(
            "  Node {:2}: predicted class {} (logits: [{}])",
            i,
            best_class,
            class_logits.join(", ")
        );
    }

    println!(
        "\nAccuracy on synthetic test graph: {:.2}%",
        correct as f64 / num_nodes as f64 * 100.0
    );

    // Print attention weights from layer 1 (first head)
    println!("\n--- Layer 1 attention weights (head 0, first 5 nodes) ---");
    let adj_data = adj.to_vec_f64().unwrap();
    let layer1_w = &model.layer1.linears[0];
    let layer1_attn = &model.layer1.attn_linears[0];

    // Recompute attention for display
    let feat_var2 = Variable::new(random_features(num_nodes, num_features));
    let wh = layer1_w.forward(&feat_var2);
    let wh_data = wh.tensor().to_vec_f64().unwrap();
    let f_out = model.layer1.out_features;

    for i in 0..5.min(num_nodes) {
        let mut neighbor_attns = Vec::new();
        for j in 0..num_nodes {
            if adj_data[i * num_nodes + j] > 0.5 {
                let mut concat_vec = vec![0.0f64; 2 * f_out];
                concat_vec[..f_out]
                    .copy_from_slice(&wh_data[i * f_out..(i + 1) * f_out]);
                concat_vec[f_out..2 * f_out]
                    .copy_from_slice(&wh_data[j * f_out..(j + 1) * f_out]);
                let cat_var = Variable::new(Tensor::from_slice(&concat_vec, &[1, 2 * f_out]));
                let e = layer1_attn.forward(&cat_var);
                let e_val = e.tensor().item().unwrap();
                let e_val = if e_val > 0.0 { e_val } else { 0.2 * e_val };
                neighbor_attns.push((j, e_val));
            }
        }
        let attn_strs: Vec<String> = neighbor_attns
            .iter()
            .map(|(j, a)| format!("{}:{:.4}", j, a))
            .collect();
        println!("  Node {}: [{}]", i, attn_strs.join(", "));
    }

    println!("\nInference complete.");
}
