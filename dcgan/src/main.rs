//! Deep Convolutional GAN (DCGAN) example.
//!
//! Implements a GAN with linear layers simulating transposed convolutions:
//! - Generator: Linear(100, 256) -> ReLU -> ... -> Linear(1024, 784) -> Tanh
//! - Discriminator: Linear(784, 512) -> ReLU -> ... -> Linear(256, 1) -> Sigmoid
//!
//! Trained on synthetic flattened 28x28 images.

use theano_nn::BCELoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use dcgan::{random_noise, synthetic_real_data, target_tensor, DCGAN};

fn main() {
    println!("=== Deep Convolutional GAN (DCGAN) ===");
    println!();

    let model = DCGAN::new();

    let g_params = model.generator.parameters();
    let d_params = model.discriminator.parameters();

    let g_param_count: usize = g_params.iter().map(|p| p.tensor().numel()).sum();
    let d_param_count: usize = d_params.iter().map(|p| p.tensor().numel()).sum();

    let mut g_optimizer = Adam::new(g_params, 2e-4).betas(0.5, 0.999);
    let mut d_optimizer = Adam::new(d_params, 2e-4).betas(0.5, 0.999);

    let bce = BCELoss::new();
    let batch_size = 64;
    let latent_dim = 100;
    let num_epochs = 20;
    let batches_per_epoch = 10;

    println!("Generator parameters: {}", g_param_count);
    println!("Discriminator parameters: {}", d_param_count);
    println!("Batch size: {}", batch_size);
    println!("Latent dimension: {}", latent_dim);
    println!("Epochs: {}", num_epochs);
    println!();

    for epoch in 1..=num_epochs {
        let mut total_d_loss = 0.0;
        let mut total_g_loss = 0.0;

        for _ in 0..batches_per_epoch {
            // ---------------------------------------------------------------
            // Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            // ---------------------------------------------------------------
            d_optimizer.zero_grad();

            // Real data
            let real_data = synthetic_real_data(batch_size);
            let real_labels = target_tensor(batch_size, 1.0);
            let d_real_output = model.discriminator.forward(&real_data);
            let d_real_loss = bce.forward(&d_real_output, &real_labels);

            // Fake data
            let noise = random_noise(batch_size, latent_dim);
            let fake_data = model.generator.forward(&noise);
            let fake_labels = target_tensor(batch_size, 0.0);
            let d_fake_output = model.discriminator.forward(&fake_data.detach());
            let d_fake_loss = bce.forward(&d_fake_output, &fake_labels);

            let d_loss_val = d_real_loss.tensor().item().unwrap()
                + d_fake_loss.tensor().item().unwrap();
            total_d_loss += d_loss_val;

            // Backward for discriminator (sum of both losses)
            let d_loss = d_real_loss.add(&d_fake_loss).unwrap();
            d_loss.backward();
            d_optimizer.step();

            // ---------------------------------------------------------------
            // Train Generator: maximize log(D(G(z)))
            // ---------------------------------------------------------------
            g_optimizer.zero_grad();

            let noise = random_noise(batch_size, latent_dim);
            let fake_data = model.generator.forward(&noise);
            let real_labels_for_g = target_tensor(batch_size, 1.0);
            let g_output = model.discriminator.forward(&fake_data);
            let g_loss = bce.forward(&g_output, &real_labels_for_g);

            let g_loss_val = g_loss.tensor().item().unwrap();
            total_g_loss += g_loss_val;

            g_loss.backward();
            g_optimizer.step();
        }

        let avg_d_loss = total_d_loss / batches_per_epoch as f64;
        let avg_g_loss = total_g_loss / batches_per_epoch as f64;

        println!(
            "Epoch [{:2}/{}]  D Loss: {:.4}  G Loss: {:.4}",
            epoch, num_epochs, avg_d_loss, avg_g_loss
        );
    }

    // Generate a sample
    println!();
    println!("Generating sample from trained generator...");
    let sample_noise = random_noise(1, latent_dim);
    let sample = model.generator.forward(&sample_noise);
    let sample_data = sample.tensor().to_vec_f64().unwrap();
    let sample_min = sample_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let sample_max = sample_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Sample stats: min={:.4}, max={:.4}, shape={:?}",
        sample_min,
        sample_max,
        sample.tensor().shape()
    );
    println!();
    println!("Training complete.");

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("dcgan_model.safetensors", bytes).unwrap();
    println!("Model saved to dcgan_model.safetensors");
}
