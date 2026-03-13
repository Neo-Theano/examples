//! MNIST RNN Example — Classify MNIST digits using a recurrent neural network.
//!
//! Treats each 28x28 image as a sequence of 28 timesteps, each with a 28-dimensional
//! input vector (one row of the image). The final hidden state is projected to 10
//! classes for digit classification.
//!
//! Architecture:
//!   LSTMCell(input_size=28, hidden_size=128)
//!   Linear(128, 10)

use theano::prelude::*;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use mnist_rnn::{accuracy, generate_batch, print_model_summary, MnistRNN};

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

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("mnist_rnn_model.safetensors", bytes).unwrap();
    println!("Model saved to mnist_rnn_model.safetensors");
}
