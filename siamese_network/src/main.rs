//! Siamese Network training example.
//!
//! Trains on synthetic pairs and saves the model to `siamese_model.safetensors`.

use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;
use siamese_network::{
    contrastive_loss, contrastive_loss_value, synthetic_pairs, SiameseNetwork,
};

fn main() {
    println!("=== Siamese Network ===");
    println!();

    let model = SiameseNetwork::new();
    let batch_size = 32;
    let num_epochs = 20;
    let batches_per_epoch = 10;
    let margin = 2.0;

    let param_count: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("Model parameters: {}", param_count);
    println!("Embedding dimension: 64");
    println!("Contrastive margin: {}", margin);
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", num_epochs);
    println!();

    let mut optimizer = Adam::new(model.parameters(), 1e-3);

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;
        let mut total_same_dist = 0.0;
        let mut total_diff_dist = 0.0;
        let mut same_count = 0;
        let mut diff_count = 0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            let (x1, x2, labels) = synthetic_pairs(batch_size);
            let (emb1, emb2) = model.forward(&x1, &x2);

            // Compute loss (with gradient graph)
            let loss = contrastive_loss(&emb1, &emb2, &labels, margin);

            // Compute display loss value
            let loss_val = contrastive_loss_value(&emb1, &emb2, &labels, margin);
            total_loss += loss_val;

            // Track average distances for same/different pairs
            let e1 = emb1.tensor().to_vec_f64().unwrap();
            let e2 = emb2.tensor().to_vec_f64().unwrap();
            let embed_dim = 64;
            for i in 0..batch_size {
                let mut dist_sq = 0.0;
                for j in 0..embed_dim {
                    let diff = e1[i * embed_dim + j] - e2[i * embed_dim + j];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                if labels[i] > 0.5 {
                    total_same_dist += dist;
                    same_count += 1;
                } else {
                    total_diff_dist += dist;
                    diff_count += 1;
                }
            }

            loss.backward();
            optimizer.step();
        }

        let avg_loss = total_loss / batches_per_epoch as f64;
        let avg_same = if same_count > 0 {
            total_same_dist / same_count as f64
        } else {
            0.0
        };
        let avg_diff = if diff_count > 0 {
            total_diff_dist / diff_count as f64
        } else {
            0.0
        };

        println!(
            "Epoch [{:2}/{}]  Loss: {:.4}  Avg Same Dist: {:.4}  Avg Diff Dist: {:.4}",
            epoch, num_epochs, avg_loss, avg_same, avg_diff
        );
    }

    // Test with a few pairs
    println!();
    println!("Testing with sample pairs...");
    let (test_x1, test_x2, test_labels) = synthetic_pairs(5);
    let (test_e1, test_e2) = model.forward(&test_x1, &test_x2);
    let e1_data = test_e1.tensor().to_vec_f64().unwrap();
    let e2_data = test_e2.tensor().to_vec_f64().unwrap();
    let embed_dim = 64;

    for i in 0..5 {
        let mut dist_sq = 0.0;
        for j in 0..embed_dim {
            let diff = e1_data[i * embed_dim + j] - e2_data[i * embed_dim + j];
            dist_sq += diff * diff;
        }
        let label_str = if test_labels[i] > 0.5 { "same" } else { "diff" };
        println!(
            "  Pair {}: label={}, distance={:.4}",
            i + 1,
            label_str,
            dist_sq.sqrt()
        );
    }

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("siamese_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to siamese_model.safetensors");
}
