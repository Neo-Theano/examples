//! GCN inference example.
//!
//! Loads a trained GCN model, creates a synthetic graph, classifies nodes,
//! and prints predictions.

use theano_autograd::Variable;
use theano_serialize::load_state_dict;

use gcn::{GCN, random_normalized_adjacency, random_features, compute_accuracy, random_labels};

fn main() {
    println!("=== GCN Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("gcn_model.safetensors")
        .expect("Model file not found. Run training first: cargo run --bin gcn");
    let sd = load_state_dict(&bytes).unwrap();
    let model = GCN::from_state_dict(&sd);
    println!("Model loaded from gcn_model.safetensors");

    // Create a synthetic graph for inference
    let num_nodes = 20;
    let num_features = 16;
    let num_classes = 5;
    let edge_prob = 0.15;

    let adj = random_normalized_adjacency(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj);
    let feat_var = Variable::new(features);

    // Run inference
    println!("\n--- Node classification on synthetic graph ---");
    println!("Graph: {} nodes, {} features", num_nodes, num_features);

    let logits = model.forward(&adj_var, &feat_var);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Print predictions for each node
    println!("\n--- Node predictions ---");
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
        if i < 10 {
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
    }
    if num_nodes > 10 {
        println!("  ... ({} more nodes)", num_nodes - 10);
    }

    let accuracy = compute_accuracy(logits.tensor(), &labels);
    println!(
        "\nAccuracy on synthetic test graph: {:.2}%",
        accuracy * 100.0
    );
    println!("\nInference complete.");
}
