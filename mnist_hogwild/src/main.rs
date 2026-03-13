//! MNIST Hogwild Example — Multi-threaded MNIST training with shared parameters.
//!
//! Demonstrates the Hogwild! asynchronous SGD pattern (Niu et al., 2011) where
//! multiple workers train on the same shared model without locking. In a real
//! implementation, workers would race on parameter updates.
//!
//! Since true shared-memory Hogwild training across threads requires unsafe in Rust
//! (or atomic operations on model parameters), this example shows the API structure
//! and simulates multi-process training sequentially.

use theano::prelude::*;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use mnist_hogwild::{generate_batch, print_model_summary, MnistCNN};

/// Simulate Hogwild training for a single worker process.
///
/// In a real Hogwild implementation, this function would:
///   1. Run in its own thread/process
///   2. Read shared parameters (without lock)
///   3. Compute gradients on local data
///   4. Write gradient updates back to shared parameters (without lock)
///
/// The key insight of Hogwild! is that for sparse problems like MNIST,
/// conflicts between concurrent updates are rare and the algorithm still converges.
fn train_worker(
    worker_id: usize,
    model: &MnistCNN,
    optimizer: &mut Adam,
    criterion: &CrossEntropyLoss,
    num_batches: usize,
    batch_size: usize,
) {
    println!("  [Worker {}] Starting training ({} batches)...", worker_id, num_batches);

    let mut total_loss = 0.0;
    for batch_idx in 0..num_batches {
        let (images, labels) = generate_batch(batch_size);
        let input = Variable::new(images);
        let target = Variable::new(labels);

        // Forward
        optimizer.zero_grad();
        let output = model.forward(&input);
        let loss = criterion.forward(&output, &target);

        // Backward + update (in real Hogwild, this writes to shared memory)
        loss.backward();
        optimizer.step();

        let loss_val = loss.tensor().item().unwrap();
        total_loss += loss_val;

        if (batch_idx + 1) % 5 == 0 {
            println!(
                "  [Worker {}] Batch [{}/{}], Loss: {:.4}",
                worker_id,
                batch_idx + 1,
                num_batches,
                loss_val
            );
        }
    }

    let avg_loss = total_loss / num_batches as f64;
    println!(
        "  [Worker {}] Finished. Average Loss: {:.4}",
        worker_id, avg_loss
    );
}

fn main() {
    println!("Neo Theano — MNIST Hogwild Example");
    println!("(Demonstrating Hogwild-style shared-parameter training API)\n");

    // Configuration
    let num_processes = 4;
    let lr = 0.001;
    let batch_size = 4;
    let batches_per_worker = 5;
    let num_epochs = 2;

    println!("Hogwild training with {} processes", num_processes);
    println!(
        "Each worker processes {} batches of size {} per epoch\n",
        batches_per_worker, batch_size
    );

    // Build the shared model
    let mut model = MnistCNN::new();
    print_model_summary(&model);

    let criterion = CrossEntropyLoss::new();

    // Training
    for epoch in 0..num_epochs {
        println!("\n--- Epoch {}/{} ---", epoch + 1, num_epochs);
        model.set_train();

        for worker_id in 0..num_processes {
            let params = model.parameters();
            let mut optimizer = Adam::new(params, lr);
            train_worker(
                worker_id,
                &model,
                &mut optimizer,
                &criterion,
                batches_per_worker,
                batch_size,
            );
        }
    }

    // Evaluation
    println!("\nEvaluating...");
    model.set_eval();
    let mut total_correct = 0.0;
    let total_samples = 20;
    let eval_batch_size = 4;
    let eval_batches = total_samples / eval_batch_size;

    for _ in 0..eval_batches {
        let (images, labels) = generate_batch(eval_batch_size);
        let input = Variable::new(images);
        let logits = model.forward(&input);

        let logits_data = logits.tensor().to_vec_f64().unwrap();
        let labels_data = labels.to_vec_f64().unwrap();
        let n = eval_batch_size;
        let c = 10;

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
                total_correct += 1.0;
            }
        }
    }

    println!(
        "Test Accuracy: {:.2}% ({}/{} correct)",
        total_correct / total_samples as f64 * 100.0,
        total_correct as usize,
        total_samples
    );
    println!("(Note: accuracy is ~10% random baseline since we use synthetic data)");

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("mnist_hogwild_model.safetensors", bytes).unwrap();
    println!("Model saved to mnist_hogwild_model.safetensors");
}
