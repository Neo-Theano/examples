//! Super-Resolution inference example.
//!
//! Loads the trained ESPCN model, upscales a synthetic 8x8 image,
//! and prints output dimensions and pixel statistics.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;

use super_resolution::SuperResolutionNet;

fn main() {
    println!("=== Super-Resolution (ESPCN) Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("super_resolution_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin super_resolution");
    let sd = load_state_dict(&bytes).unwrap();
    let model = SuperResolutionNet::from_state_dict(&sd);
    println!("Model loaded from super_resolution_model.safetensors");
    println!("Upscale factor: {}x", model.upscale_factor);

    // Create a synthetic 8x8 input image
    let lr_h = 8;
    let lr_w = 8;
    let input_data: Vec<f64> = (0..lr_h * lr_w)
        .map(|i| (i as f64) / (lr_h * lr_w) as f64)
        .collect();
    let input = Variable::new(Tensor::from_slice(&input_data, &[1, 1, lr_h, lr_w]));

    println!("\n--- Upscaling synthetic image ---");
    println!("Input shape:  {:?}", input.tensor().shape());

    let output = model.forward(&input);
    let output_shape = output.tensor().shape().to_vec();
    let output_data = output.tensor().to_vec_f64().unwrap();

    println!("Output shape: {:?}", output_shape);
    println!(
        "Expected output: [1, 1, {}, {}]",
        lr_h * model.upscale_factor,
        lr_w * model.upscale_factor
    );

    // Pixel statistics
    let pixel_mean: f64 = output_data.iter().sum::<f64>() / output_data.len() as f64;
    let pixel_std: f64 = (output_data
        .iter()
        .map(|v| (v - pixel_mean).powi(2))
        .sum::<f64>()
        / output_data.len() as f64)
        .sqrt();
    let pixel_min = output_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let pixel_max = output_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n--- Output Pixel Statistics ---");
    println!("  Mean:  {:.4}", pixel_mean);
    println!("  Std:   {:.4}", pixel_std);
    println!("  Min:   {:.4}", pixel_min);
    println!("  Max:   {:.4}", pixel_max);
    println!("  Total pixels: {}", output_data.len());

    // Show a few output pixel values
    println!("\n--- First 10 output pixels ---");
    for (i, &val) in output_data.iter().take(10).enumerate() {
        println!("  pixel[{}] = {:.4}", i, val);
    }

    println!("\nInference complete.");
}
