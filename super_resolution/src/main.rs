//! Super-Resolution example using a sub-pixel convolution network.
//!
//! Implements a simplified ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
//! for 2x upscaling:
//! - Conv2d(1, 64, 5, padding=2) -> ReLU
//! - Conv2d(64, 32, 3, padding=1) -> ReLU
//! - Conv2d(32, 4, 3, padding=1)   (4 = upscale_factor^2 for pixel shuffle)
//! - Pixel shuffle reshape to simulate 2x upscaling
//!
//! Trained with MSELoss on synthetic low-res / high-res image pairs.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Conv2d, Module, MSELoss};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct SuperResolutionNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    upscale_factor: usize,
}

impl SuperResolutionNet {
    fn new(upscale_factor: usize) -> Self {
        // Conv2d(in_channels, out_channels, kernel_size) with padding
        let conv1 = Conv2d::with_options(1, 64, (5, 5), (1, 1), (2, 2), true);
        let conv2 = Conv2d::with_options(64, 32, (3, 3), (1, 1), (1, 1), true);
        // Output channels = upscale_factor^2 for pixel shuffle
        let out_channels = upscale_factor * upscale_factor;
        let conv3 = Conv2d::with_options(32, out_channels, (3, 3), (1, 1), (1, 1), true);

        Self {
            conv1,
            conv2,
            conv3,
            upscale_factor,
        }
    }

    /// Forward pass. Input: [N, 1, H, W] -> Output: [N, 1, H*upscale, W*upscale]
    fn forward(&self, x: &Variable) -> Variable {
        let h = self.conv1.forward(x);
        let h = h.relu().unwrap();
        let h = self.conv2.forward(&h);
        let h = h.relu().unwrap();
        let h = self.conv3.forward(&h);

        // Pixel shuffle: reshape [N, r^2, H, W] -> [N, 1, H*r, W*r]
        pixel_shuffle(&h, self.upscale_factor)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Pixel Shuffle
// ---------------------------------------------------------------------------

/// Rearranges elements in a tensor of shape [N, C*r^2, H, W] to [N, C, H*r, W*r].
/// This is a simplified version that assumes C=1 after shuffle.
fn pixel_shuffle(input: &Variable, upscale_factor: usize) -> Variable {
    let shape = input.tensor().shape();
    let n = shape[0];
    let c = shape[1]; // should be upscale_factor^2
    let h = shape[2];
    let w = shape[3];
    let r = upscale_factor;

    assert_eq!(c, r * r, "Input channels must be upscale_factor^2");

    let data = input.tensor().to_vec_f64().unwrap();
    let out_h = h * r;
    let out_w = w * r;
    let mut output = vec![0.0f64; n * 1 * out_h * out_w];

    for batch in 0..n {
        for ih in 0..h {
            for iw in 0..w {
                for rh in 0..r {
                    for rw in 0..r {
                        let in_c = rh * r + rw;
                        let in_idx = batch * c * h * w + in_c * h * w + ih * w + iw;
                        let oh = ih * r + rh;
                        let ow = iw * r + rw;
                        let out_idx = batch * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = data[in_idx];
                    }
                }
            }
        }
    }

    Variable::new(Tensor::from_slice(&output, &[n, 1, out_h, out_w]))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate synthetic low-res and high-res image pairs.
/// Low-res: [batch, 1, h, w], High-res: [batch, 1, h*r, w*r]
fn synthetic_image_pairs(
    batch_size: usize,
    lr_h: usize,
    lr_w: usize,
    upscale_factor: usize,
) -> (Variable, Variable) {
    let mut rng = rand::thread_rng();
    let hr_h = lr_h * upscale_factor;
    let hr_w = lr_w * upscale_factor;

    // Generate high-res images (smooth random patterns)
    let hr_numel = batch_size * hr_h * hr_w;
    let hr_data: Vec<f64> = (0..hr_numel).map(|_| rng.gen::<f64>()).collect();

    // Downsample to create low-res (simple average pooling)
    let lr_numel = batch_size * lr_h * lr_w;
    let mut lr_data = vec![0.0f64; lr_numel];

    for b in 0..batch_size {
        for i in 0..lr_h {
            for j in 0..lr_w {
                let mut sum = 0.0;
                for di in 0..upscale_factor {
                    for dj in 0..upscale_factor {
                        let hi = i * upscale_factor + di;
                        let hj = j * upscale_factor + dj;
                        sum += hr_data[b * hr_h * hr_w + hi * hr_w + hj];
                    }
                }
                lr_data[b * lr_h * lr_w + i * lr_w + j] =
                    sum / (upscale_factor * upscale_factor) as f64;
            }
        }
    }

    let lr = Variable::new(Tensor::from_slice(&lr_data, &[batch_size, 1, lr_h, lr_w]));
    let hr = Variable::new(Tensor::from_slice(&hr_data, &[batch_size, 1, hr_h, hr_w]));

    (lr, hr)
}

/// Compute PSNR (Peak Signal-to-Noise Ratio) given MSE value.
fn psnr_from_mse(mse: f64) -> f64 {
    if mse < 1e-12 {
        100.0 // Perfect reconstruction
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
}
