//! SNLI inference example.
//!
//! Loads a trained SNLI model and classifies a synthetic premise/hypothesis pair.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;

use snli::{SNLIClassifier, VOCAB_SIZE, NUM_CLASSES, SEQ_LEN};

fn main() {
    println!("=== SNLI Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("snli_model.safetensors")
        .expect("Model file not found. Run training first: cargo run --bin snli");
    let sd = load_state_dict(&bytes).unwrap();
    let model = SNLIClassifier::from_state_dict(&sd);
    println!("Model loaded from snli_model.safetensors");

    // Create synthetic premise and hypothesis
    let mut rng = rand::thread_rng();
    let premise_data: Vec<f64> = (0..SEQ_LEN)
        .map(|_| rng.gen_range(1..VOCAB_SIZE) as f64)
        .collect();
    let hypothesis_data: Vec<f64> = (0..SEQ_LEN)
        .map(|_| rng.gen_range(1..VOCAB_SIZE) as f64)
        .collect();

    let premise = Variable::new(Tensor::from_slice(&premise_data, &[1, SEQ_LEN]));
    let hypothesis = Variable::new(Tensor::from_slice(&hypothesis_data, &[1, SEQ_LEN]));

    // Run classification
    println!("\n--- Classifying premise/hypothesis pair ---");
    let premise_tokens: Vec<usize> = premise_data.iter().map(|&x| x as usize).collect();
    let hypothesis_tokens: Vec<usize> = hypothesis_data.iter().map(|&x| x as usize).collect();
    println!("  Premise tokens:    {:?}", premise_tokens);
    println!("  Hypothesis tokens: {:?}", hypothesis_tokens);

    let logits = model.forward(&premise, &hypothesis);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Compute softmax probabilities
    let max_val = logits_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits_data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_vals.iter().sum();
    let probs: Vec<f64> = exp_vals.iter().map(|&x| x / sum_exp).collect();

    let class_names = ["Entailment", "Contradiction", "Neutral"];
    println!("\n--- Class probabilities ---");
    let mut best_class = 0;
    let mut best_prob = 0.0;
    for c in 0..NUM_CLASSES {
        let name = if c < class_names.len() {
            class_names[c]
        } else {
            "Unknown"
        };
        println!("  {}: {:.4} (logit: {:.4})", name, probs[c], logits_data[c]);
        if probs[c] > best_prob {
            best_prob = probs[c];
            best_class = c;
        }
    }

    println!(
        "\nPrediction: {} (confidence: {:.2}%)",
        class_names[best_class],
        best_prob * 100.0
    );
    println!("\nInference complete.");
}
