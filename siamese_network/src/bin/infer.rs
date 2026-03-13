//! Siamese Network inference example.
//!
//! Loads a trained SiameseNetwork, computes embeddings for two sample images,
//! and computes the distance between them.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;
use siamese_network::SiameseNetwork;

fn main() {
    println!("=== Siamese Network Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("siamese_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin siamese_network");
    let sd = load_state_dict(&bytes).unwrap();
    let model = SiameseNetwork::from_state_dict(&sd);
    println!("Model loaded from siamese_model.safetensors");

    // Create two sample images
    let mut rng = rand::thread_rng();
    let img1_data: Vec<f64> = (0..784).map(|_| rng.gen::<f64>()).collect();
    let img2_data: Vec<f64> = (0..784).map(|_| rng.gen::<f64>()).collect();

    let img1 = Variable::new(Tensor::from_slice(&img1_data, &[1, 784]));
    let img2 = Variable::new(Tensor::from_slice(&img2_data, &[1, 784]));

    // Compute embeddings
    let (emb1, emb2) = model.forward(&img1, &img2);
    let e1 = emb1.tensor().to_vec_f64().unwrap();
    let e2 = emb2.tensor().to_vec_f64().unwrap();

    println!();
    println!("Embedding 1 (first 5): {:?}", &e1[..5].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("Embedding 2 (first 5): {:?}", &e2[..5].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());

    // Compute Euclidean distance
    let embed_dim = 64;
    let mut dist_sq = 0.0;
    for j in 0..embed_dim {
        let diff = e1[j] - e2[j];
        dist_sq += diff * diff;
    }
    let distance = dist_sq.sqrt();

    println!();
    println!("Euclidean distance: {:.4}", distance);
    println!("Distance squared:   {:.4}", dist_sq);

    // Test with a similar image (noisy copy of img1)
    println!();
    println!("--- Testing with similar image (noisy copy) ---");
    let img3_data: Vec<f64> = img1_data.iter().map(|&v| v + rng.gen::<f64>() * 0.1 - 0.05).collect();
    let img3 = Variable::new(Tensor::from_slice(&img3_data, &[1, 784]));
    let emb3 = model.forward_one(&img3);
    let e3 = emb3.tensor().to_vec_f64().unwrap();

    let mut similar_dist_sq = 0.0;
    for j in 0..embed_dim {
        let diff = e1[j] - e3[j];
        similar_dist_sq += diff * diff;
    }
    println!("Distance (original vs noisy copy): {:.4}", similar_dist_sq.sqrt());

    println!("\nInference complete.");
}
