//! Language Translation inference example.
//!
//! Loads a trained translation model and translates a synthetic source sequence.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;

use language_translation::{
    TranslationModel, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN,
};

fn main() {
    println!("=== Language Translation Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("translation_model.safetensors").expect(
        "Model file not found. Run training first: cargo run --bin language_translation",
    );
    let sd = load_state_dict(&bytes).unwrap();
    let model = TranslationModel::from_state_dict(&sd);
    println!("Model loaded from translation_model.safetensors");

    // Create a synthetic source sequence
    let mut rng = rand::thread_rng();
    let sample_src: Vec<f64> = (0..SRC_SEQ_LEN)
        .map(|_| rng.gen_range(1..SRC_VOCAB_SIZE) as f64)
        .collect();
    let sample_tgt_in: Vec<f64> = (0..TGT_SEQ_LEN)
        .map(|_| rng.gen_range(1..TGT_VOCAB_SIZE) as f64)
        .collect();

    let src_var = Variable::new(Tensor::from_slice(&sample_src, &[1, SRC_SEQ_LEN]));
    let tgt_var = Variable::new(Tensor::from_slice(&sample_tgt_in, &[1, TGT_SEQ_LEN]));

    // Run translation
    println!("\n--- Translating synthetic source sequence ---");
    let logits = model.forward(&src_var, &tgt_var);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Greedy decode: argmax at each position
    let mut predicted_tokens = Vec::new();
    for t in 0..TGT_SEQ_LEN {
        let offset = t * TGT_VOCAB_SIZE;
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for v in 0..TGT_VOCAB_SIZE {
            if logits_data[offset + v] > best_val {
                best_val = logits_data[offset + v];
                best_idx = v;
            }
        }
        predicted_tokens.push(best_idx);
    }

    let input_tokens: Vec<usize> = sample_src.iter().map(|&x| x as usize).collect();
    println!("  Source tokens:    {:?}", input_tokens);
    println!("  Predicted tokens: {:?}", predicted_tokens);

    // Print logits for first few positions
    println!("\n--- Output logits (first 3 positions, top-5 tokens) ---");
    for t in 0..3.min(TGT_SEQ_LEN) {
        let offset = t * TGT_VOCAB_SIZE;
        let mut indexed: Vec<(usize, f64)> = (0..TGT_VOCAB_SIZE)
            .map(|v| (v, logits_data[offset + v]))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<String> = indexed[..5]
            .iter()
            .map(|(idx, val)| format!("{}:{:.4}", idx, val))
            .collect();
        println!("  Position {}: [{}]", t, top5.join(", "));
    }

    println!("\nInference complete.");
}
