//! Reinforcement Learning inference example.
//!
//! Loads a trained policy network, takes actions on sample states,
//! and prints action probabilities.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_serialize::load_state_dict;

use reinforcement_learning::{PolicyNetwork, STATE_DIM, NUM_ACTIONS};

fn main() {
    println!("=== REINFORCE Policy Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("rl_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin reinforcement_learning");
    let sd = load_state_dict(&bytes).unwrap();
    let policy = PolicyNetwork::from_state_dict(&sd);
    println!("Model loaded from rl_model.safetensors");
    println!("State dim: {}, Num actions: {}", STATE_DIM, NUM_ACTIONS);

    // Sample states for inference
    let sample_states: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.5, -0.5, -0.5],
    ];

    println!("\n--- Action Probabilities for Sample States ---");
    for (i, state) in sample_states.iter().enumerate() {
        let state_var = Variable::new(Tensor::from_slice(state, &[1, STATE_DIM]));
        let action_probs = policy.forward(&state_var);
        let probs_data = action_probs.tensor().to_vec_f64().unwrap();

        // Find the best action (greedy)
        let best_action = probs_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let probs_str: Vec<String> = probs_data.iter().map(|p| format!("{:.4}", p)).collect();
        println!(
            "  State {}: {:?} -> probs: [{}] -> best action: {}",
            i + 1,
            state,
            probs_str.join(", "),
            best_action
        );
    }

    println!("\nInference complete.");
}
