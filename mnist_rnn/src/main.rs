//! MNIST RNN Example — Classify MNIST digits using a recurrent neural network.
//!
//! Treats each 28x28 image as a sequence of 28 timesteps, each with a 28-dimensional
//! input vector (one row of the image). The final hidden state is projected to 10
//! classes for digit classification.
//!
//! Architecture:
//!   LSTMCell(input_size=28, hidden_size=128)
//!   Linear(128, 10)

use rand::Rng;
use theano::prelude::*;
use theano_nn::{CrossEntropyLoss, LSTMCell, Linear, Module};
use theano_optim::{Adam, Optimizer};

/// RNN model for MNIST classification.
///
/// Processes each 28x28 image as 28 timesteps of 28-dim vectors,
/// then uses the final hidden state for classification.
struct MnistRNN {
    lstm: LSTMCell,
    fc: Linear,
    hidden_size: usize,
    seq_len: usize,
    input_size: usize,
}

impl MnistRNN {
    fn new(input_size: usize, hidden_size: usize, num_classes: usize, seq_len: usize) -> Self {
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
    fn forward(&self, x: &Variable) -> Variable {
        let shape = x.tensor().shape().to_vec();
        let batch = shape[0];

        // Reshape [batch, 1, 28, 28] -> extract rows as sequence
        // We need to get each row: [batch, 28] for each of 28 timesteps
        let flat_data = x.tensor().to_vec_f64().unwrap();

        // Initialize hidden and cell states
        let mut h = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));
        let mut c = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));

        // Process each row as a timestep
        for t in 0..self.seq_len {
            // Extract row t from each image: [batch, input_size]
            let mut row_data = vec![0.0f64; batch * self.input_size];
            for b in 0..batch {
                for j in 0..self.input_size {
                    // flat layout: batch * 1 * 28 * 28
                    // pixel at (b, 0, t, j) = b * 784 + t * 28 + j
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

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.lstm.parameters());
        params.extend(self.fc.parameters());
        params
    }
}

/// Print the model architecture and total parameter count.
fn print_model_summary(model: &MnistRNN) {
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
fn generate_batch(batch_size: usize) -> (Tensor, Tensor) {
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
fn accuracy(logits: &Tensor, labels: &Tensor) -> f64 {
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

fn main() {
    println!("Neo Theano — MNIST RNN Example");
    println!("(Using LSTM to classify MNIST digits as sequences)\n");

    // Hyperparameters
    let input_size = 28; // each row of the image
    let hidden_size = 128;
    let num_classes = 10;
    let seq_len = 28; // 28 rows per image
    let lr = 0.001;
    let num_epochs = 3;
    let batch_size = 4;
    let train_batches = 10;
    let test_batches = 5;

    // Build model
    let model = MnistRNN::new(input_size, hidden_size, num_classes, seq_len);
    print_model_summary(&model);

    // Optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(params, lr);

    // Loss function
    let criterion = CrossEntropyLoss::new();

    // Training loop
    println!("Training...");
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..train_batches {
            let (images, labels) = generate_batch(batch_size);
            let input = Variable::new(images);
            let target = Variable::new(labels);

            // Forward
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = criterion.forward(&output, &target);

            // Backward
            loss.backward();
            optimizer.step();

            let loss_val = loss.tensor().item().unwrap();
            epoch_loss += loss_val;

            if batch_idx % 5 == 0 {
                println!(
                    "  Epoch [{}/{}], Batch [{}/{}], Loss: {:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    train_batches,
                    loss_val
                );
            }
        }

        let avg_loss = epoch_loss / train_batches as f64;
        println!(
            "  Epoch [{}/{}] Average Loss: {:.4}",
            epoch + 1,
            num_epochs,
            avg_loss
        );
    }

    // Evaluation
    println!("\nEvaluating...");
    let mut total_correct = 0.0;
    let mut total_samples = 0;

    for _ in 0..test_batches {
        let (images, labels) = generate_batch(batch_size);
        let input = Variable::new(images);
        let output = model.forward(&input);

        let acc = accuracy(output.tensor(), &labels);
        total_correct += acc * batch_size as f64;
        total_samples += batch_size;
    }

    let test_acc = total_correct / total_samples as f64;
    println!(
        "Test Accuracy: {:.2}% ({}/{} correct)",
        test_acc * 100.0,
        total_correct as usize,
        total_samples
    );
    println!("(Note: accuracy is ~10% random baseline since we use synthetic data)");
}
