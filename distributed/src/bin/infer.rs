//! Distributed model inference example.
//!
//! Loads a trained SimpleModel and classifies synthetic input.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;
use distributed::SimpleModel;

fn main() {
    println!("=== Distributed Model Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("distributed_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin distributed");
    let sd = load_state_dict(&bytes).unwrap();
    let model = SimpleModel::from_state_dict(&sd);
    println!("Model loaded from distributed_model.safetensors");

    // Create synthetic input
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..128).map(|_| rng.gen::<f64>()).collect();
    let input = Variable::new(Tensor::from_slice(&data, &[1, 128]));

    // Run forward pass
    let logits = model.forward(&input);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    println!();
    println!("Input shape: [1, 128]");
    println!("Output logits (10 classes):");
    for (i, logit) in logits_data.iter().enumerate() {
        println!("  Class {}: {:.4}", i, logit);
    }

    // Find predicted class
    let (pred_class, pred_val) = logits_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!();
    println!("Predicted class: {} (logit: {:.4})", pred_class, pred_val);

    println!("\nInference complete.");
}
