//! MNIST Forward-Forward inference example.
//!
//! Loads a trained Forward-Forward network and classifies synthetic images
//! by testing all possible label overlays and picking the one with highest goodness.

use rand::Rng;
use theano_serialize::load_state_dict;

use mnist_forward_forward::FFNetwork;

fn main() {
    println!("=== MNIST Forward-Forward Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("mnist_ff_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin mnist_forward_forward");
    let sd = load_state_dict(&bytes).unwrap();
    let network = FFNetwork::from_state_dict(&sd);
    println!("Model loaded from mnist_ff_model.safetensors");
    println!("  Network has {} layers", network.layers.len());

    // Create synthetic images and classify them
    let input_dim = 784;
    let num_classes = 10;
    println!("\n--- Classifying synthetic images ---");
    let mut rng = rand::thread_rng();

    for sample_idx in 0..5 {
        let image: Vec<f64> = (0..input_dim).map(|_| rng.gen::<f64>()).collect();
        let images = vec![image];

        let predictions = network.predict(&images, num_classes);
        let predicted_class = predictions[0];

        println!(
            "  Sample {}: Predicted digit = {}",
            sample_idx + 1,
            predicted_class
        );
    }

    println!("\n(Note: predictions are random-quality since the model was trained on synthetic data)");
    println!("Inference complete.");
}
