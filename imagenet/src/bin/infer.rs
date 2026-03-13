//! ResNet-18 inference example.
//!
//! Loads a saved ResNet-18 model and classifies a synthetic 224x224 RGB image.

use theano_autograd::Variable;
use theano_nn::Module;
use theano_serialize::load_state_dict;

use imagenet::{ResNet18, random_tensor};

fn main() {
    println!("=== ResNet-18 Inference ===\n");

    // Load saved model
    let bytes = std::fs::read("resnet18_model.safetensors")
        .expect("failed to read resnet18_model.safetensors — run training first");
    let sd = load_state_dict(&bytes).expect("failed to parse state dict");
    println!("Loaded state dict with {} tensors", sd.len());

    let num_classes = 1000;
    let model = ResNet18::from_state_dict(&sd, num_classes);
    println!("ResNet-18 model reconstructed.\n");

    // Create a synthetic 224x224 RGB image
    let image = random_tensor(&[1, 3, 224, 224]);
    let input = Variable::new(image);

    // Forward pass
    let logits = model.forward(&input);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Find top-5 class indices
    let mut indexed: Vec<(usize, f64)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top-5 predictions:");
    for (rank, (class_idx, score)) in indexed.iter().take(5).enumerate() {
        println!("  #{}: class {} (score: {:.4})", rank + 1, class_idx, score);
    }

    println!("\nInference complete.");
}
