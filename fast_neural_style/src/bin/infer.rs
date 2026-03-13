//! Fast Neural Style Transfer inference example.
//!
//! Loads a trained TransformerNet and stylizes a synthetic 16x16 RGB image.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;
use fast_neural_style::TransformerNet;

fn main() {
    println!("=== Fast Neural Style Transfer Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("style_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin fast_neural_style");
    let sd = load_state_dict(&bytes).unwrap();
    let transformer = TransformerNet::from_state_dict(&sd);
    println!("Model loaded from style_model.safetensors");

    // Create a synthetic 16x16 RGB image
    let img_h = 16;
    let img_w = 16;
    let mut rng = rand::thread_rng();
    let numel = 1 * 3 * img_h * img_w;
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    let input = Variable::new(Tensor::from_slice(&data, &[1, 3, img_h, img_w]));

    println!("Input shape:  {:?}", input.tensor().shape());

    // Run forward pass (stylize)
    let output = transformer.forward(&input);
    let output_shape = output.tensor().shape().to_vec();
    println!("Output shape: {:?}", output_shape);

    // Print output statistics
    let out_data = output.tensor().to_vec_f64().unwrap();
    let out_min = out_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let out_max = out_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let out_mean = out_data.iter().sum::<f64>() / out_data.len() as f64;
    let out_std = (out_data.iter().map(|v| (v - out_mean).powi(2)).sum::<f64>()
        / out_data.len() as f64)
        .sqrt();

    println!();
    println!("Output statistics:");
    println!("  Min:  {:.4}", out_min);
    println!("  Max:  {:.4}", out_max);
    println!("  Mean: {:.4}", out_mean);
    println!("  Std:  {:.4}", out_std);

    // Print per-channel stats
    let c = output_shape[1];
    let h = output_shape[2];
    let w = output_shape[3];
    println!();
    println!("Per-channel statistics:");
    for ch in 0..c {
        let channel_data: Vec<f64> = (0..h * w)
            .map(|i| out_data[ch * h * w + i])
            .collect();
        let ch_mean = channel_data.iter().sum::<f64>() / channel_data.len() as f64;
        let ch_min = channel_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let ch_max = channel_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ch_name = match ch {
            0 => "R",
            1 => "G",
            2 => "B",
            _ => "?",
        };
        println!(
            "  Channel {} ({}): mean={:.4}, min={:.4}, max={:.4}",
            ch, ch_name, ch_mean, ch_min, ch_max
        );
    }

    println!("\nInference complete.");
}
