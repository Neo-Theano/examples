//! Time Sequence Prediction model definitions.
//!
//! LSTM-based model for predicting sine wave sequences:
//! - LSTMCell(1, 32) for sequential processing
//! - Linear(32, 1) for output prediction
//! - Teacher forcing during training, free-running at inference

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{LSTMCell, Linear, Module};

// Hyperparameters
pub const HIDDEN_SIZE: usize = 32;
pub const INPUT_SIZE: usize = 1;
pub const OUTPUT_SIZE: usize = 1;
pub const SEQ_LEN: usize = 50;
pub const NUM_SEQUENCES: usize = 20;
pub const NUM_EPOCHS: usize = 10;
pub const LEARNING_RATE: f64 = 0.005;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SineLSTM {
    pub lstm_cell: LSTMCell,
    pub output_linear: Linear,
}

impl SineLSTM {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            lstm_cell: LSTMCell::new(INPUT_SIZE, hidden_size),
            output_linear: Linear::new(hidden_size, OUTPUT_SIZE),
        }
    }

    /// Forward with teacher forcing.
    /// input_seq: [batch, seq_len] -- the input values at each timestep
    /// Returns: predictions [batch, seq_len] -- predicted next value at each step
    pub fn forward_teacher_forcing(&self, input_seq: &Variable) -> Variable {
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
    pub fn predict(&self, start_val: &[f64], steps: usize) -> Vec<f64> {
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

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.lstm_cell.parameters();
        params.extend(self.output_linear.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        // LSTMCell parameters: [w_ih, w_hh, b_ih, b_hh]
        let lstm_params = self.lstm_cell.parameters();
        sd.insert("lstm.w_ih".to_string(), lstm_params[0].tensor().clone());
        sd.insert("lstm.w_hh".to_string(), lstm_params[1].tensor().clone());
        sd.insert("lstm.b_ih".to_string(), lstm_params[2].tensor().clone());
        sd.insert("lstm.b_hh".to_string(), lstm_params[3].tensor().clone());
        // Linear parameters
        for (name, param) in self.output_linear.named_parameters() {
            sd.insert(format!("fc.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            lstm_cell: LSTMCell::from_tensors(
                sd["lstm.w_ih"].clone(),
                sd["lstm.w_hh"].clone(),
                sd["lstm.b_ih"].clone(),
                sd["lstm.b_hh"].clone(),
            ),
            output_linear: Linear::from_tensors(
                sd["fc.weight"].clone(),
                Some(sd["fc.bias"].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate sine wave data: sin(x + phase) for different phases.
/// Returns (inputs, targets) where targets are shifted by 1 step.
/// inputs: [num_sequences, seq_len], targets: [num_sequences, seq_len]
pub fn generate_sine_data(num_sequences: usize, seq_len: usize) -> (Vec<f64>, Vec<f64>) {
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
