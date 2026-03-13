//! MNIST Forward-Forward Example — Hinton's Forward-Forward Algorithm (2022).
//!
//! Instead of backpropagation, each layer is trained independently with a local
//! "goodness" objective. Positive data (real images with correct labels) should
//! produce high goodness, while negative data (images with wrong labels) should
//! produce low goodness.
//!
//! Goodness = sum of squared activations in the layer.
//!
//! Each layer has its own optimizer and is trained to:
//!   - Increase goodness above a threshold for positive data
//!   - Decrease goodness below a threshold for negative data
//!
//! This is a fundamentally different training paradigm from backpropagation.

use rand::Rng;
use theano_optim::Adam;
use theano_serialize::save_state_dict;

use mnist_forward_forward::{print_model_summary, FFNetwork};

fn main() {
    println!("Neo Theano — MNIST Forward-Forward Algorithm Example");
    println!("(Hinton 2022: Training without backpropagation)\n");

    // Hyperparameters
    let input_dim = 784; // 28*28 flattened
    let num_classes = 10;
    let threshold = 2.0;
    let lr = 0.001;
    let num_epochs = 5;
    let batch_size = 8;
    let train_batches = 10;
    let test_batches = 5;

    // Build network: 784 -> 500 -> 500 -> 500
    let layer_sizes = vec![input_dim, 500, 500, 500];
    let network = FFNetwork::new(&layer_sizes, threshold);
    print_model_summary(&network);

    // Create per-layer optimizers (each layer trained independently!)
    let mut optimizers: Vec<Adam> = network
        .layers
        .iter()
        .map(|layer| Adam::new(layer.parameters(), lr))
        .collect();

    // Training loop
    println!("Training with Forward-Forward algorithm...");
    println!("(Each layer trained independently - no backprop through the network!)\n");

    let mut rng = rand::thread_rng();

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..train_batches {
            // Generate synthetic batch
            let images: Vec<Vec<f64>> = (0..batch_size)
                .map(|_| (0..input_dim).map(|_| rng.gen::<f64>()).collect())
                .collect();
            let labels: Vec<usize> = (0..batch_size)
                .map(|_| rng.gen_range(0..num_classes))
                .collect();

            let loss = network.train_epoch(&mut optimizers, &images, &labels, num_classes);
            epoch_loss += loss;

            if batch_idx % 5 == 0 {
                println!(
                    "  Epoch [{}/{}], Batch [{}/{}], Loss: {:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    train_batches,
                    loss
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
    let mut total_correct = 0;
    let mut total_samples = 0;

    for _ in 0..test_batches {
        let images: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| (0..input_dim).map(|_| rng.gen::<f64>()).collect())
            .collect();
        let labels: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..num_classes))
            .collect();

        let predictions = network.predict(&images, num_classes);

        for i in 0..batch_size {
            if predictions[i] == labels[i] {
                total_correct += 1;
            }
        }
        total_samples += batch_size;
    }

    let test_acc = total_correct as f64 / total_samples as f64;
    println!(
        "Test Accuracy: {:.2}% ({}/{} correct)",
        test_acc * 100.0,
        total_correct,
        total_samples
    );
    println!("(Note: accuracy is ~10% random baseline since we use synthetic data)");
    println!("\nKey insight: No gradient flows between layers!");
    println!("Each layer is trained with its own local objective (goodness).");

    // Save the trained model
    let sd = network.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("mnist_ff_model.safetensors", bytes).unwrap();
    println!("Model saved to mnist_ff_model.safetensors");
}
