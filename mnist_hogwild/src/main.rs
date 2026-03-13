//! MNIST Hogwild Example — Multi-threaded MNIST training with shared parameters.
//!
//! Demonstrates the Hogwild! asynchronous SGD pattern (Niu et al., 2011) where
//! multiple workers train on the same shared model without locking. In a real
//! implementation, workers would race on parameter updates.
//!
//! Since true shared-memory Hogwild training across threads requires unsafe in Rust
//! (or atomic operations on model parameters), this example shows the API structure
//! and simulates multi-process training sequentially.

use rand::Rng;
use theano::prelude::*;
use theano_nn::{
    Conv2d, CrossEntropyLoss, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU,
};
use theano_optim::{Adam, Optimizer};

/// CNN model for MNIST — same architecture as the basic mnist example.
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
        let x = self.conv1.forward(x);
        let x = x.relu().unwrap();
        let x = self.conv2.forward(&x);
        let x = x.relu().unwrap();
        let x = self.pool.forward(&x);
        let x = self.dropout1.forward(&x);
        let x = self.flatten.forward(&x);
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
    let total_params: usize = model
        .parameters()
        .iter()
        .map(|p| p.tensor().numel())
        .sum();
    println!("Model: MNIST CNN ({} parameters)", total_params);
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
    // In real Hogwild!, model parameters live in shared memory (e.g., via mmap or
    // shared Arc<AtomicF64> parameters). All processes read/write without locks.
    let mut model = MnistCNN::new();
    print_model_summary(&model);

    let criterion = CrossEntropyLoss::new();

    // Training
    for epoch in 0..num_epochs {
        println!("\n--- Epoch {}/{} ---", epoch + 1, num_epochs);
        model.set_train();

        // In real Hogwild!, each worker would be spawned as a separate thread:
        //
        //   let model = Arc::new(model);  // shared model
        //   let handles: Vec<_> = (0..num_processes).map(|worker_id| {
        //       let model = model.clone();
        //       std::thread::spawn(move || {
        //           train_worker(worker_id, &model, ...);
        //       })
        //   }).collect();
        //   for h in handles { h.join().unwrap(); }
        //
        // Here we simulate sequentially to show the pattern:

        for worker_id in 0..num_processes {
            // Each worker gets its own optimizer that updates the shared model params.
            // In real Hogwild, gradient updates race on shared memory.
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
}
