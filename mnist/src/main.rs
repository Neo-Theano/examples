//! MNIST CNN Example — Classic convolutional neural network for digit classification.
//!
//! Architecture mirrors the PyTorch MNIST example:
//!   Conv2d(1,32,3) -> ReLU -> Conv2d(32,64,3) -> ReLU -> MaxPool2d(2)
//!   -> Dropout(0.25) -> Flatten -> Linear(9216,128) -> ReLU
//!   -> Dropout(0.5) -> Linear(128,10)
//!
//! Uses synthetic random data that mimics real MNIST shapes (28x28 grayscale images).

use rand::Rng;
use theano::prelude::*;
use theano_nn::{
    Conv2d, CrossEntropyLoss, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU,
};
use theano_optim::{Adam, Optimizer};

/// CNN model for MNIST digit classification.
struct MnistCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: MaxPool2d,
    dropout1: Dropout,
    flatten: Flatten,
    fc1: Linear,
    relu: ReLU,
    dropout2: Dropout,
    fc2: Linear,
}

impl MnistCNN {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            pool: MaxPool2d::new(2),
            dropout1: Dropout::new(0.25),
            flatten: Flatten::new(),
            fc1: Linear::new(9216, 128),
            relu: ReLU,
            dropout2: Dropout::new(0.5),
            fc2: Linear::new(128, 10),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        // Conv block 1: Conv2d(1,32,3) -> ReLU
        let x = self.conv1.forward(x);
        let x = x.relu().unwrap();

        // Conv block 2: Conv2d(32,64,3) -> ReLU -> MaxPool2d(2)
        let x = self.conv2.forward(&x);
        let x = x.relu().unwrap();
        let x = self.pool.forward(&x);

        // Dropout -> Flatten
        let x = self.dropout1.forward(&x);
        let x = self.flatten.forward(&x);

        // FC block: Linear(9216,128) -> ReLU -> Dropout(0.5) -> Linear(128,10)
        let x = self.fc1.forward(&x);
        let x = self.relu.forward(&x);
        let x = self.dropout2.forward(&x);
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn set_eval(&mut self) {
        self.dropout1.eval();
        self.dropout2.eval();
    }

    fn set_train(&mut self) {
        self.dropout1.train();
        self.dropout2.train();
    }
}

/// Print the model architecture and total parameter count.
fn print_model_summary(model: &MnistCNN) {
    println!("=== MNIST CNN Architecture ===");
    println!("  Conv2d(1, 32, kernel_size=3)    -> ReLU");
    println!("  Conv2d(32, 64, kernel_size=3)   -> ReLU -> MaxPool2d(2)");
    println!("  Dropout(0.25)");
    println!("  Flatten");
    println!("  Linear(9216, 128)               -> ReLU");
    println!("  Dropout(0.5)");
    println!("  Linear(128, 10)");
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
/// Returns (images: [N, 1, 28, 28], labels: [N]).
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

/// Compute accuracy: fraction of correct predictions.
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
    println!("Neo Theano — MNIST CNN Example");
    println!("(Using synthetic random data)\n");

    // Build model
    let mut model = MnistCNN::new();
    print_model_summary(&model);

    // Hyperparameters
    let lr = 0.001;
    let num_epochs = 3;
    let batch_size = 4; // Small batch for demo (CNN on CPU is slow)
    let train_batches = 10;
    let test_batches = 5;

    // Optimizer
    let params = model.parameters();
    let mut optimizer = Adam::new(params, lr);

    // Loss function
    let criterion = CrossEntropyLoss::new();

    // Training loop
    println!("Training...");
    for epoch in 0..num_epochs {
        model.set_train();
        let mut epoch_loss = 0.0;

        for batch_idx in 0..train_batches {
            let (images, labels) = generate_batch(batch_size);
            let input = Variable::new(images);
            let target = Variable::new(labels);

            // Forward pass
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = criterion.forward(&output, &target);

            // Backward pass
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

    // Evaluation loop
    println!("\nEvaluating...");
    model.set_eval();
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
    println!(
        "(Note: accuracy is ~10% random baseline since we use synthetic data)"
    );
}
