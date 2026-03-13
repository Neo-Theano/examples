//! MNIST RNN model definitions.
//!
//! Treats each 28x28 image as a sequence of 28 timesteps, each with a 28-dimensional
//! input vector (one row of the image). The final hidden state is projected to 10
//! classes for digit classification.
//!
//! Architecture:
//!   LSTMCell(input_size=28, hidden_size=128)
//!   Linear(128, 10)

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{LSTMCell, Linear, Module};

/// RNN model for MNIST classification.
///
/// Processes each 28x28 image as 28 timesteps of 28-dim vectors,
/// then uses the final hidden state for classification.
pub struct MnistRNN {
    pub lstm: LSTMCell,
    pub fc: Linear,
    pub hidden_size: usize,
    pub seq_len: usize,
    pub input_size: usize,
}

impl MnistRNN {
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize, seq_len: usize) -> Self {
        Self {
            lstm: LSTMCell::new(input_size, hidden_size),
            fc: Linear::new(hidden_size, num_classes),
            hidden_size,
            seq_len,
            input_size,
        }
    }

    /// Forward pass: process image row-by-row through the LSTM.
    ///
    /// Input shape: [batch, 1, 28, 28] (MNIST image)
    /// Output shape: [batch, 10] (class logits)
    pub fn forward(&self, x: &Variable) -> Variable {
        let shape = x.tensor().shape().to_vec();
        let batch = shape[0];

        // Reshape [batch, 1, 28, 28] -> extract rows as sequence
        let flat_data = x.tensor().to_vec_f64().unwrap();

        // Initialize hidden and cell states
        let mut h = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));
        let mut c = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));

        // Process each row as a timestep
        for t in 0..self.seq_len {
            let mut row_data = vec![0.0f64; batch * self.input_size];
            for b in 0..batch {
                for j in 0..self.input_size {
                    row_data[b * self.input_size + j] = flat_data[b * 784 + t * 28 + j];
                }
            }

            let row_input =
                Variable::new(Tensor::from_slice(&row_data, &[batch, self.input_size]));

            let (new_h, new_c) = self.lstm.forward_cell(&row_input, &h, &c);
            h = new_h;
            c = new_c;
        }

        // Use final hidden state for classification
        self.fc.forward(&h)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.lstm.parameters());
        params.extend(self.fc.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // LSTM parameters: [w_ih, w_hh, b_ih, b_hh]
        let lstm_params = self.lstm.parameters();
        sd.insert("lstm.w_ih".to_string(), lstm_params[0].tensor().clone());
        sd.insert("lstm.w_hh".to_string(), lstm_params[1].tensor().clone());
        sd.insert("lstm.b_ih".to_string(), lstm_params[2].tensor().clone());
        sd.insert("lstm.b_hh".to_string(), lstm_params[3].tensor().clone());

        // Store hyperparameters as 1-element tensors
        sd.insert("hidden_size".to_string(), Tensor::from_slice(&[self.hidden_size as f64], &[1]));
        sd.insert("seq_len".to_string(), Tensor::from_slice(&[self.seq_len as f64], &[1]));
        sd.insert("input_size".to_string(), Tensor::from_slice(&[self.input_size as f64], &[1]));

        // FC parameters
        for (name, param) in self.fc.named_parameters() {
            sd.insert(format!("fc.{name}"), param.tensor().clone());
        }

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        let hidden_size = sd["hidden_size"].to_vec_f64().unwrap()[0] as usize;
        let seq_len = sd["seq_len"].to_vec_f64().unwrap()[0] as usize;
        let input_size = sd["input_size"].to_vec_f64().unwrap()[0] as usize;

        Self {
            lstm: LSTMCell::from_tensors(
                sd["lstm.w_ih"].clone(),
                sd["lstm.w_hh"].clone(),
                sd["lstm.b_ih"].clone(),
                sd["lstm.b_hh"].clone(),
            ),
            fc: Linear::from_tensors(
                sd["fc.weight"].clone(),
                Some(sd["fc.bias"].clone()),
            ),
            hidden_size,
            seq_len,
            input_size,
        }
    }
}

/// Print the model architecture and total parameter count.
pub fn print_model_summary(model: &MnistRNN) {
    println!("=== MNIST RNN Architecture ===");
    println!("  Input: 28x28 image treated as 28 timesteps of 28-dim vectors");
    println!("  LSTMCell(input_size=28, hidden_size={})", model.hidden_size);
    println!("  Linear({}, 10)", model.hidden_size);
    println!("==============================");

    let total_params: usize = model
        .parameters()
        .iter()
        .map(|p| p.tensor().numel())
        .sum();
    println!("Total trainable parameters: {}", total_params);
    println!();
}

/// Generate a batch of synthetic MNIST-like data.
pub fn generate_batch(batch_size: usize) -> (Tensor, Tensor) {
    let mut rng = rand::thread_rng();
    let img_numel = batch_size * 1 * 28 * 28;
    let img_data: Vec<f64> = (0..img_numel).map(|_| rng.gen::<f64>()).collect();
    let label_data: Vec<f64> = (0..batch_size)
        .map(|_| rng.gen_range(0..10) as f64)
        .collect();

    let images = Tensor::from_slice(&img_data, &[batch_size, 1, 28, 28]);
    let labels = Tensor::from_slice(&label_data, &[batch_size]);
    (images, labels)
}

/// Compute accuracy.
pub fn accuracy(logits: &Tensor, labels: &Tensor) -> f64 {
    let n = logits.shape()[0];
    let c = logits.shape()[1];
    let logits_data = logits.to_vec_f64().unwrap();
    let labels_data = labels.to_vec_f64().unwrap();

    let mut correct = 0;
    for i in 0..n {
        let mut best_class = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..c {
            let v = logits_data[i * c + j];
            if v > best_val {
                best_val = v;
                best_class = j;
            }
        }
        if best_class == labels_data[i] as usize {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}
