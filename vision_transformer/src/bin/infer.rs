//! Vision Transformer inference example.
//!
//! Loads a saved ViT model and classifies a synthetic 32x32 RGB image.

use theano_autograd::Variable;
use theano_nn::Module;
use theano_serialize::load_state_dict;

use vision_transformer::{ViT, random_images};

fn main() {
    println!("=== Vision Transformer (ViT) Inference ===\n");

    // Load saved model
    let bytes = std::fs::read("vit_model.safetensors")
        .expect("failed to read vit_model.safetensors — run training first");
    let sd = load_state_dict(&bytes).expect("failed to parse state dict");
    println!("Loaded state dict with {} tensors", sd.len());

    let img_channels = 3;
    let img_size = 32;
    let patch_size = 4;
    let num_heads = 4;
    let num_blocks = 4;

    let model = ViT::from_state_dict(&sd, img_channels, img_size, patch_size, num_heads, num_blocks);
    println!("ViT model reconstructed.\n");

    // Create a synthetic 32x32 RGB image
    let image = random_images(1, img_channels, img_size, img_size);
    let input = Variable::new(image);

    // Forward pass
    let logits = model.forward(&input);
    let logits_data = logits.tensor().to_vec_f64().unwrap();
    let num_classes = logits_data.len();

    // Find top predictions
    let mut indexed: Vec<(usize, f64)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k = num_classes.min(5);
    println!("Top-{} predictions:", top_k);
    for (rank, (class_idx, score)) in indexed.iter().take(top_k).enumerate() {
        println!("  #{}: class {} (score: {:.4})", rank + 1, class_idx, score);
    }

    println!("\nInference complete.");
}
