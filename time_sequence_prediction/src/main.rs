//! Time Sequence Prediction — LSTM for predicting sine wave sequences.
//!
//! Generates synthetic sine wave data with different phases.
//! Model: LSTMCell processing one step at a time, Linear(hidden, 1) for prediction.
//! Uses teacher forcing during training.
//! Prints MSE loss per epoch and shows predicted vs actual for the last sequence.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{LSTMCell, Linear, Module, MSELoss};
use theano_optim::{Adam, Optimizer};

// Hyperparameters
const HIDDEN_SIZE: usize = 32;
const INPUT_SIZE: usize = 1;
const OUTPUT_SIZE: usize = 1;
const SEQ_LEN: usize = 50;
const NUM_SEQUENCES: usize = 20;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.005;

/// Sine wave LSTM model.
struct SineLSTM {
    lstm_cell: LSTMCell,
    output_linear: Linear,
}

impl SineLSTM {
    fn new(hidden_size: usize) -> Self {
        Self {
            lstm_cell: LSTMCell::new(INPUT_SIZE, hidden_size),
            output_linear: Linear::new(hidden_size, OUTPUT_SIZE),
        }
    }

    /// Forward with teacher forcing.
    /// input_seq: [batch, seq_len] — the input values at each timestep
    /// Returns: predictions [batch, seq_len] — predicted next value at each step
    fn forward_teacher_forcing(&self, input_seq: &Variable) -> Variable {
        let shape = input_seq.tensor().shape().to_vec();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let data = input_seq.tensor().to_vec_f64().unwrap();

        let mut h = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut c = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));

        let mut all_preds = Vec::with_capacity(batch_size * seq_len);

        for t in 0..seq_len {
            // Extract input at timestep t: [batch, 1]
            let mut step_data = vec![0.0f64; batch_size];
            for b in 0..batch_size {
                step_data[b] = data[b * seq_len + t];
            }
            let step_input = Variable::new(Tensor::from_slice(&step_data, &[batch_size, INPUT_SIZE]));

            // LSTM step
            let (new_h, new_c) = self.lstm_cell.forward_cell(&step_input, &h, &c);
            h = new_h;
            c = new_c;

            // Predict from hidden state
            let pred = self.output_linear.forward(&h); // [batch, 1]
            let pred_data = pred.tensor().to_vec_f64().unwrap();

            for b in 0..batch_size {
                all_preds.push(pred_data[b]);
            }
        }

        Variable::new(Tensor::from_slice(&all_preds, &[batch_size, seq_len]))
    }

    /// Free-running prediction (no teacher forcing).
    /// start_val: initial input value per sequence [batch, 1]
    /// steps: number of steps to predict
    fn predict(&self, start_val: &[f64], steps: usize) -> Vec<f64> {
        let batch_size = start_val.len();
        let mut h = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut c = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut current = start_val.to_vec();
        let mut predictions = Vec::with_capacity(batch_size * steps);

        for _ in 0..steps {
            let input = Variable::new(Tensor::from_slice(&current, &[batch_size, INPUT_SIZE]));
            let (new_h, new_c) = self.lstm_cell.forward_cell(&input, &h, &c);
            h = new_h;
            c = new_c;

            let pred = self.output_linear.forward(&h);
            let pred_data = pred.tensor().to_vec_f64().unwrap();

            for b in 0..batch_size {
                predictions.push(pred_data[b]);
                current[b] = pred_data[b]; // feed prediction back as input
            }
        }

        predictions
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.lstm_cell.parameters();
        params.extend(self.output_linear.parameters());
        params
    }
}

/// Generate sine wave data: sin(x + phase) for different phases.
/// Returns (inputs, targets) where targets are shifted by 1 step.
/// inputs: [num_sequences, seq_len], targets: [num_sequences, seq_len]
fn generate_sine_data(num_sequences: usize, seq_len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::with_capacity(num_sequences * seq_len);
    let mut targets = Vec::with_capacity(num_sequences * seq_len);

    for _ in 0..num_sequences {
        let phase: f64 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let freq: f64 = rng.gen_range(0.8..1.2);

        for t in 0..seq_len {
            let x = t as f64 * 0.1;
            let y_in = (freq * x + phase).sin();
            let y_out = (freq * (x + 0.1) + phase).sin(); // shifted by one step
            inputs.push(y_in);
            targets.push(y_out);
        }
    }

    (inputs, targets)
}

fn main() {
    println!("Time Sequence Prediction — LSTM Sine Wave Example");
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
            "Epoch [{:2}/{}] — MSE Loss: {:.6}",
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

    println!();
    println!("Training complete.");
}
