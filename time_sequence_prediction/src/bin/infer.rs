//! Time Sequence Prediction inference example.
//!
//! Loads a trained SineLSTM model and predicts future sine wave values.

use theano_serialize::load_state_dict;
use time_sequence_prediction::{SineLSTM, SEQ_LEN};

fn main() {
    println!("=== Time Sequence Prediction Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("time_seq_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin time_sequence_prediction");
    let sd = load_state_dict(&bytes).unwrap();
    let model = SineLSTM::from_state_dict(&sd);
    println!("Model loaded from time_seq_model.safetensors");

    // Generate a sine wave sequence to predict
    let test_phase = 0.5;
    let test_freq = 1.0;
    let actual: Vec<f64> = (0..SEQ_LEN)
        .map(|t| (test_freq * (t as f64 * 0.1 + 0.1) + test_phase).sin())
        .collect();

    let start_val = (test_freq * 0.0 + test_phase).sin();
    let predicted = model.predict(&[start_val], SEQ_LEN);

    println!();
    println!("Predicted vs Actual (free-running prediction):");
    println!("-----------------------------------------------");
    println!("{:<8} {:>10} {:>10} {:>10}", "Step", "Predicted", "Actual", "Error");
    let display_steps = 10.min(SEQ_LEN);
    for t in 0..display_steps {
        let error = (predicted[t] - actual[t]).abs();
        println!(
            "{:<8} {:>10.4} {:>10.4} {:>10.4}",
            t + 1,
            predicted[t],
            actual[t],
            error
        );
    }
    if SEQ_LEN > display_steps {
        println!("... ({} more steps)", SEQ_LEN - display_steps);
    }

    // Compute overall MSE
    let mse: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / SEQ_LEN as f64;
    println!();
    println!("Overall MSE: {:.6}", mse);

    println!("\nInference complete.");
}
