//! VAE inference example.
//!
//! Loads a trained VAE model and:
//! 1. Encodes a sample image to get latent representation
//! 2. Generates new images by sampling from the latent space

use rand_distr::{Distribution, Normal};
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;
use vae::VAE;

fn main() {
    println!("=== VAE Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("vae_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin vae");
    let sd = load_state_dict(&bytes).unwrap();
    let vae = VAE::from_state_dict(&sd);
    println!("Model loaded from vae_model.safetensors");

    // 1. Encode a sample image and inspect latent representation
    println!("\n--- Encoding a sample image ---");
    let sample = vae::synthetic_batch(1);
    let (recon, mu, logvar) = vae.forward(&sample);

    let mu_data = mu.tensor().to_vec_f64().unwrap();
    let logvar_data = logvar.tensor().to_vec_f64().unwrap();
    println!("Latent mu (first 5):    {:?}", &mu_data[..5]);
    println!("Latent logvar (first 5): {:?}", &logvar_data[..5]);

    let recon_data = recon.tensor().to_vec_f64().unwrap();
    println!(
        "Reconstruction (first 10 pixels): {:?}",
        &recon_data[..10]
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );

    // 2. Generate new images by sampling from latent space
    println!("\n--- Generating from latent space ---");
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    for i in 0..3 {
        let z_data: Vec<f64> = (0..20).map(|_| dist.sample(&mut rng)).collect();
        let z = Variable::new(Tensor::from_slice(&z_data, &[1, 20]));
        let generated = vae.decoder.forward(&z);
        let gen_data = generated.tensor().to_vec_f64().unwrap();
        let pixel_mean: f64 = gen_data.iter().sum::<f64>() / gen_data.len() as f64;
        let pixel_std: f64 = (gen_data.iter().map(|v| (v - pixel_mean).powi(2)).sum::<f64>()
            / gen_data.len() as f64)
            .sqrt();
        println!(
            "  Sample {}: mean pixel={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
            i + 1,
            pixel_mean,
            pixel_std,
            gen_data.iter().cloned().fold(f64::INFINITY, f64::min),
            gen_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }

    println!("\nInference complete.");
}
