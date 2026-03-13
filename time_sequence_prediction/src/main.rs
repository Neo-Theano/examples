//! Time Sequence Prediction -- LSTM for predicting sine wave sequences.
//!
//! Trains on synthetic sine wave data and saves the model to `time_seq_model.safetensors`.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::MSELoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;
use time_sequence_prediction::{
    generate_sine_data, SineLSTM, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_SEQUENCES, SEQ_LEN,
};

fn main() {
    println!("Time Sequence Prediction -- LSTM Sine Wave Example");
    println!("====================================================");
    println!("Hidden: {HIDDEN_SIZE}, Seq len: {SEQ_LEN}, Sequences: {NUM_SEQUENCES}");
    println!();

    let model = SineLSTM::new(HIDDEN_SIZE);
    let criterion = MSELoss::new();
    let params = model.parameters();
    println!("Model parameters: {}", params.len());

    let mut optimizer = Adam::new(params, LEARNING_RATE);

    for epoch in 0..NUM_EPOCHS {
        // Generate fresh data each epoch
        let (input_data, target_data) = generate_sine_data(NUM_SEQUENCES, SEQ_LEN);

        let inputs = Variable::new(Tensor::from_slice(
            &input_data,
            &[NUM_SEQUENCES, SEQ_LEN],
        ));
        let targets = Variable::new(Tensor::from_slice(
            &target_data,
            &[NUM_SEQUENCES, SEQ_LEN],
        ));

        // Forward with teacher forcing
        let predictions = model.forward_teacher_forcing(&inputs);

        // Compute MSE loss
        let loss = criterion.forward(&predictions, &targets);
        let loss_val = loss.tensor().item().unwrap();

        // Backward and update
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        println!(
            "Epoch [{:2}/{}] -- MSE Loss: {:.6}",
            epoch + 1,
            NUM_EPOCHS,
            loss_val
        );
    }

    // Show predicted vs actual for a single test sequence
    println!();
    println!("Predicted vs Actual (last test sequence, free-running):");
    println!("-------------------------------------------------------");

    let test_phase = 0.5;
    let test_freq = 1.0;
    let actual: Vec<f64> = (0..SEQ_LEN)
        .map(|t| (test_freq * (t as f64 * 0.1 + 0.1) + test_phase).sin())
        .collect();

    let start_val = (test_freq * 0.0 + test_phase).sin();
    let predicted = model.predict(&[start_val], SEQ_LEN);

    let display_steps = 10.min(SEQ_LEN);
    println!("{:<8} {:>10} {:>10}", "Step", "Predicted", "Actual");
    for t in 0..display_steps {
        println!(
            "{:<8} {:>10.4} {:>10.4}",
            t + 1,
            predicted[t],
            actual[t]
        );
    }
    if SEQ_LEN > display_steps {
        println!("... ({} more steps)", SEQ_LEN - display_steps);
    }

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("time_seq_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to time_seq_model.safetensors");
}
