//! Word Language Model inference example.
//!
//! Loads a saved LSTM language model, predicts next tokens for a synthetic
//! sequence, and prints perplexity and top predictions.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::CrossEntropyLoss;
use theano_serialize::load_state_dict;

use word_language_model::{
    LSTMLanguageModel, generate_batch, VOCAB_SIZE, SEQ_LEN,
};

fn main() {
    println!("=== Word Language Model (LSTM) Inference ===\n");

    // Load saved model
    let bytes = std::fs::read("word_lm_model.safetensors")
        .expect("failed to read word_lm_model.safetensors — run training first");
    let sd = load_state_dict(&bytes).expect("failed to parse state dict");
    println!("Loaded state dict with {} tensors", sd.len());

    let model = LSTMLanguageModel::from_state_dict(&sd);
    println!("LSTM language model reconstructed.");
    println!("  Vocab size:   {}", model.vocab_size);
    println!("  Hidden size:  {}\n", model.hidden_size);

    // Generate a synthetic input sequence (batch_size=1)
    let batch_size = 1;
    let (input_data, target_data) = generate_batch(batch_size, SEQ_LEN, VOCAB_SIZE);

    let input = Variable::new(Tensor::from_slice(&input_data, &[batch_size, SEQ_LEN]));
    let target = Variable::new(Tensor::from_slice(&target_data, &[batch_size * SEQ_LEN]));

    // Forward pass
    let logits = model.forward_seq(&input);

    // Compute perplexity
    let criterion = CrossEntropyLoss::new();
    let loss = criterion.forward(&logits, &target);
    let loss_val = loss.tensor().item().unwrap();
    let perplexity = loss_val.exp();
    println!("Sequence perplexity: {:.2}", perplexity);

    // Show top-5 predictions for the last timestep
    let logits_data = logits.tensor().to_vec_f64().unwrap();
    let last_step_start = (SEQ_LEN - 1) * VOCAB_SIZE;
    let last_logits = &logits_data[last_step_start..last_step_start + VOCAB_SIZE];

    let mut indexed: Vec<(usize, f64)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop-5 next-token predictions (last timestep):");
    for (rank, (token_id, score)) in indexed.iter().take(5).enumerate() {
        println!("  #{}: token {} (score: {:.4})", rank + 1, token_id, score);
    }

    println!("\nInference complete.");
}
