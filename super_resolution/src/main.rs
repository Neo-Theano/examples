//! Super-Resolution example using a sub-pixel convolution network.
//!
//! Implements a simplified ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
//! for 2x upscaling. Trained with MSELoss on synthetic low-res / high-res image pairs.

use theano_nn::MSELoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use super_resolution::{psnr_from_mse, synthetic_image_pairs, SuperResolutionNet};

fn main() {
    println!("=== Super-Resolution (ESPCN) ===");
    println!();

    let upscale_factor = 2;
    let lr_h = 8;
    let lr_w = 8;
    let batch_size = 4;
    let num_epochs = 20;
    let batches_per_epoch = 10;

    let model = SuperResolutionNet::new(upscale_factor);

    let param_count: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("Upscale factor: {}x", upscale_factor);
    println!(
        "Input size: {}x{} -> Output size: {}x{}",
        lr_h,
        lr_w,
        lr_h * upscale_factor,
        lr_w * upscale_factor
    );
    println!("Model parameters: {}", param_count);
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", num_epochs);
    println!();

    let mut optimizer = Adam::new(model.parameters(), 1e-3);
    let mse_loss = MSELoss::new();

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;
        let mut total_psnr = 0.0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            let (lr_images, hr_images) = synthetic_image_pairs(batch_size, lr_h, lr_w, upscale_factor);

            let sr_images = model.forward(&lr_images);

            let loss = mse_loss.forward(&sr_images, &hr_images);
            let loss_val = loss.tensor().item().unwrap();
            total_loss += loss_val;
            total_psnr += psnr_from_mse(loss_val);

            loss.backward();
            optimizer.step();
        }

        let avg_loss = total_loss / batches_per_epoch as f64;
        let avg_psnr = total_psnr / batches_per_epoch as f64;

        println!(
            "Epoch [{:2}/{}]  MSE Loss: {:.6}  PSNR: {:.2} dB",
            epoch, num_epochs, avg_loss, avg_psnr
        );
    }

    // Test on a single image
    println!();
    println!("Testing on a single image...");
    let (test_lr, test_hr) = synthetic_image_pairs(1, lr_h, lr_w, upscale_factor);
    let test_sr = model.forward(&test_lr);
    let test_mse = mse_loss
        .forward(&test_sr, &test_hr)
        .tensor()
        .item()
        .unwrap();
    println!(
        "Test MSE: {:.6}  PSNR: {:.2} dB",
        test_mse,
        psnr_from_mse(test_mse)
    );
    println!(
        "Input shape:  {:?}",
        test_lr.tensor().shape()
    );
    println!(
        "Output shape: {:?}",
        test_sr.tensor().shape()
    );
    println!();
    println!("Training complete.");

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("super_resolution_model.safetensors", bytes).unwrap();
    println!("Model saved to super_resolution_model.safetensors");
}
