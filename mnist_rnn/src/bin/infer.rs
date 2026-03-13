//! MNIST RNN inference example.
//!
//! Loads a trained MNIST RNN model and classifies synthetic 28x28 images
//! by processing them row-by-row as sequences.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;

use mnist_rnn::MnistRNN;

fn main() {
    println!("=== MNIST RNN Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("mnist_rnn_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin mnist_rnn");
    let sd = load_state_dict(&bytes).unwrap();
    let model = MnistRNN::from_state_dict(&sd);
    println!("Model loaded from mnist_rnn_model.safetensors");
    println!(
        "  LSTM hidden_size={}, seq_len={}, input_size={}",
        model.hidden_size, model.seq_len, model.input_size
    );

    // Create synthetic 28x28 images and classify them
    println!("\n--- Classifying synthetic images (row-by-row as sequences) ---");
    let mut rng = rand::thread_rng();

    for sample_idx in 0..3 {
        let img_data: Vec<f64> = (0..1 * 1 * 28 * 28).map(|_| rng.gen::<f64>()).collect();
        let image = Tensor::from_slice(&img_data, &[1, 1, 28, 28]);
        let input = Variable::new(image);

        let output = model.forward(&input);
        let logits = output.tensor().to_vec_f64().unwrap();

        // Compute softmax probabilities
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits.iter().map(|v| (v - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        let probs: Vec<f64> = exp_logits.iter().map(|v| v / sum_exp).collect();

        // Find top predictions
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n  Sample {}:", sample_idx + 1);
        println!("    Top predictions:");
        for (class, prob) in indexed.iter().take(3) {
            println!("      Digit {}: {:.4} ({:.1}%)", class, prob, prob * 100.0);
        }
    }

    println!("\nInference complete.");
}
