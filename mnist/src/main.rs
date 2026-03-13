//! MNIST CNN Example — Classic convolutional neural network for digit classification.
//!
//! Architecture mirrors the PyTorch MNIST example:
//!   Conv2d(1,32,3) -> ReLU -> Conv2d(32,64,3) -> ReLU -> MaxPool2d(2)
//!   -> Dropout(0.25) -> Flatten -> Linear(9216,128) -> ReLU
//!   -> Dropout(0.5) -> Linear(128,10)
//!
//! Uses synthetic random data that mimics real MNIST shapes (28x28 grayscale images).

use theano::prelude::*;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use mnist::{accuracy, generate_batch, print_model_summary, MnistCNN};

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

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("mnist_model.safetensors", bytes).unwrap();
    println!("Model saved to mnist_model.safetensors");
}
