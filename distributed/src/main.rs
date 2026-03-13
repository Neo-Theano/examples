//! Distributed Data Parallel (DDP) Training Example
//!
//! Demonstrates the distributed training API with a DistributedDataParallel
//! wrapper. Trains a SimpleModel and saves to `distributed_model.safetensors`.

use std::sync::Arc;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_distributed::{
    DistributedDataParallel, ProcessGroup, DistBackend,
    all_reduce, broadcast, barrier, CollReduceOp,
};
use theano_nn::CrossEntropyLoss;
use theano_optim::{SGD, Optimizer};
use theano_serialize::save_state_dict;
use distributed::SimpleModel;

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------
fn random_tensor(shape: &[usize]) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Tensor::from_slice(&data, shape)
}

fn random_labels(batch_size: usize, num_classes: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..batch_size)
        .map(|_| rng.gen_range(0..num_classes) as f64)
        .collect();
    Tensor::from_slice(&data, &[batch_size])
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    println!("=== Distributed Data Parallel (DDP) Training Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Initialise process group (single-process mock)
    // -----------------------------------------------------------------------
    let pg = Arc::new(ProcessGroup::new(0, 1, DistBackend::Gloo));
    println!(
        "Process group initialised: rank={}, world_size={}, backend={:?}",
        pg.rank(),
        pg.world_size(),
        pg.backend()
    );

    // -----------------------------------------------------------------------
    // 2. Build model
    // -----------------------------------------------------------------------
    let model = SimpleModel::new();

    let num_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("Model parameters: {}", num_params);

    // -----------------------------------------------------------------------
    // 3. Wrap model with DDP
    // -----------------------------------------------------------------------
    let ddp = DistributedDataParallel::new(Arc::clone(&pg));
    println!(
        "DDP wrapper created (broadcast_buffers=true, bucket_size=25MB)"
    );

    // -----------------------------------------------------------------------
    // 4. Broadcast initial parameters from rank 0
    // -----------------------------------------------------------------------
    println!("\nBroadcasting initial parameters from rank 0...");
    for param in model.parameters() {
        let _synced = broadcast(param.tensor(), 0, &pg);
    }
    println!("Parameters synchronised across {} processes.", pg.world_size());

    // -----------------------------------------------------------------------
    // 5. Training loop
    // -----------------------------------------------------------------------
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01).momentum(0.9);
    let batch_size = 32;
    let num_classes = 10;
    let num_epochs = 5;

    println!("\n--- Training ---");
    for epoch in 0..num_epochs {
        let data = random_tensor(&[batch_size, 128]);
        let labels = random_labels(batch_size, num_classes);

        optimizer.zero_grad();

        let input = Variable::new(data);
        let target = Variable::new(labels);
        let logits = model.forward(&input);
        let loss = criterion.forward(&logits, &target);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();

        // Synchronise gradients across all ranks via all-reduce
        let gradients: Vec<Tensor> = model
            .parameters()
            .iter()
            .filter_map(|p| p.grad())
            .collect();
        let synced_grads = ddp.sync_gradients(&gradients);

        // Show that gradients are synchronised
        let grad_norm: f64 = synced_grads
            .iter()
            .map(|g| {
                let d = g.to_vec_f64().unwrap();
                d.iter().map(|x| x * x).sum::<f64>()
            })
            .sum::<f64>()
            .sqrt();

        optimizer.step();

        // Barrier: ensure all ranks are in sync
        barrier(&pg);

        println!(
            "Epoch [{}/{}]  Rank: {}  World Size: {}  Loss: {:.4}  Grad Norm: {:.4}  [synchronised]",
            epoch + 1,
            num_epochs,
            pg.rank(),
            pg.world_size(),
            loss_val,
            grad_norm,
        );
    }

    // -----------------------------------------------------------------------
    // 6. Demonstrate collective operations
    // -----------------------------------------------------------------------
    println!("\n--- Collective Operations Demo ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);

    let reduced = all_reduce(&tensor, CollReduceOp::Sum, &pg);
    println!(
        "all_reduce(Sum):   input={:?}  output={:?}",
        tensor.to_vec_f64().unwrap(),
        reduced.to_vec_f64().unwrap()
    );

    let bcast = broadcast(&tensor, 0, &pg);
    println!(
        "broadcast(rank=0): input={:?}  output={:?}",
        tensor.to_vec_f64().unwrap(),
        bcast.to_vec_f64().unwrap()
    );

    barrier(&pg);
    println!("barrier():         all ranks synchronised");

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("distributed_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to distributed_model.safetensors");
}
