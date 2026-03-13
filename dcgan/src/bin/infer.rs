//! DCGAN inference example.
//!
//! Loads the trained generator, generates images from random noise,
//! and prints pixel statistics.

use theano_serialize::load_state_dict;

use dcgan::{random_noise, DCGAN};

fn main() {
    println!("=== DCGAN Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("dcgan_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin dcgan");
    let sd = load_state_dict(&bytes).unwrap();
    let model = DCGAN::from_state_dict(&sd);
    println!("Model loaded from dcgan_model.safetensors");

    // Generate images from random noise
    let latent_dim = 100;
    println!("\n--- Generating images from random noise ---");
    for i in 0..5 {
        let noise = random_noise(1, latent_dim);
        let generated = model.generator.forward(&noise);
        let gen_data = generated.tensor().to_vec_f64().unwrap();

        let pixel_mean: f64 = gen_data.iter().sum::<f64>() / gen_data.len() as f64;
        let pixel_std: f64 = (gen_data.iter().map(|v| (v - pixel_mean).powi(2)).sum::<f64>()
            / gen_data.len() as f64)
            .sqrt();
        let pixel_min = gen_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let pixel_max = gen_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "  Sample {}: shape={:?}, mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
            i + 1,
            generated.tensor().shape(),
            pixel_mean,
            pixel_std,
            pixel_min,
            pixel_max,
        );
    }

    // Test discriminator on generated samples
    println!("\n--- Discriminator scores on generated samples ---");
    for i in 0..3 {
        let noise = random_noise(1, latent_dim);
        let generated = model.generator.forward(&noise);
        let score = model.discriminator.forward(&generated);
        let score_val = score.tensor().item().unwrap();
        println!(
            "  Sample {}: D(G(z)) = {:.4}",
            i + 1,
            score_val,
        );
    }

    println!("\nInference complete.");
}
