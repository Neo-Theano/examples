//! VAE training example.
//!
//! Trains on synthetic data and saves the model to `vae_model.safetensors`.

use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;
use vae::{bce_reconstruction_loss, kl_divergence, synthetic_batch, VAE};

fn main() {
    println!("=== Variational Autoencoder (VAE) ===");
    println!();

    let vae = VAE::new();
    let mut optimizer = Adam::new(vae.parameters(), 1e-3);

    let batch_size = 64;
    let num_epochs = 20;
    let batches_per_epoch = 10;

    println!(
        "Model parameters: {}",
        vae.parameters()
            .iter()
            .map(|p| p.tensor().numel())
            .sum::<usize>()
    );
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", num_epochs);
    println!();

    for epoch in 1..=num_epochs {
        let mut total_recon_loss = 0.0;
        let mut total_kl_loss = 0.0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            let x = synthetic_batch(batch_size);
            let (recon, mu, logvar) = vae.forward(&x);

            let recon_loss = bce_reconstruction_loss(&recon, &x);
            let kl_loss = kl_divergence(&mu, &logvar);

            // Total loss = reconstruction + KL
            let loss = recon_loss.add(&kl_loss).unwrap();
            loss.backward();
            optimizer.step();

            total_recon_loss += recon_loss.tensor().item().unwrap();
            total_kl_loss += kl_loss.tensor().item().unwrap();
        }

        let avg_recon = total_recon_loss / batches_per_epoch as f64;
        let avg_kl = total_kl_loss / batches_per_epoch as f64;
        let avg_total = avg_recon + avg_kl;

        println!(
            "Epoch [{:2}/{}]  Total Loss: {:.4}  Recon Loss: {:.4}  KL Divergence: {:.4}",
            epoch, num_epochs, avg_total, avg_recon, avg_kl
        );
    }

    // Save the trained model
    let sd = vae.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("vae_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to vae_model.safetensors");
}
