//! Super-Resolution (ESPCN) model definitions.
//!
//! Implements a simplified ESPCN for 2x upscaling:
//! - Conv2d(1, 64, 5, padding=2) -> ReLU
//! - Conv2d(64, 32, 3, padding=1) -> ReLU
//! - Conv2d(32, 4, 3, padding=1)   (4 = upscale_factor^2 for pixel shuffle)
//! - Pixel shuffle reshape to simulate 2x upscaling

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Conv2d, Module};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SuperResolutionNet {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub conv3: Conv2d,
    pub upscale_factor: usize,
}

impl SuperResolutionNet {
    pub fn new(upscale_factor: usize) -> Self {
        let conv1 = Conv2d::with_options(1, 64, (5, 5), (1, 1), (2, 2), true);
        let conv2 = Conv2d::with_options(64, 32, (3, 3), (1, 1), (1, 1), true);
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
    pub fn forward(&self, x: &Variable) -> Variable {
        let h = self.conv1.forward(x);
        let h = h.relu().unwrap();
        let h = self.conv2.forward(&h);
        let h = h.relu().unwrap();
        let h = self.conv3.forward(&h);

        // Pixel shuffle: reshape [N, r^2, H, W] -> [N, 1, H*r, W*r]
        pixel_shuffle(&h, self.upscale_factor)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        let conv1_params = self.conv1.parameters();
        sd.insert("conv1.weight".to_string(), conv1_params[0].tensor().clone());
        sd.insert("conv1.bias".to_string(), conv1_params[1].tensor().clone());

        let conv2_params = self.conv2.parameters();
        sd.insert("conv2.weight".to_string(), conv2_params[0].tensor().clone());
        sd.insert("conv2.bias".to_string(), conv2_params[1].tensor().clone());

        let conv3_params = self.conv3.parameters();
        sd.insert("conv3.weight".to_string(), conv3_params[0].tensor().clone());
        sd.insert("conv3.bias".to_string(), conv3_params[1].tensor().clone());
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        // Hardcoded architecture: conv1(1,64,5,stride=1,pad=2), conv2(64,32,3,stride=1,pad=1), conv3(32,4,3,stride=1,pad=1)
        let upscale_factor = 2;
        Self {
            conv1: Conv2d::from_tensors(
                sd["conv1.weight"].clone(),
                Some(sd["conv1.bias"].clone()),
                (1, 1),
                (2, 2),
            ),
            conv2: Conv2d::from_tensors(
                sd["conv2.weight"].clone(),
                Some(sd["conv2.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
            conv3: Conv2d::from_tensors(
                sd["conv3.weight"].clone(),
                Some(sd["conv3.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
            upscale_factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel Shuffle
// ---------------------------------------------------------------------------

/// Rearranges elements in a tensor of shape [N, C*r^2, H, W] to [N, C, H*r, W*r].
/// This is a simplified version that assumes C=1 after shuffle.
pub fn pixel_shuffle(input: &Variable, upscale_factor: usize) -> Variable {
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
pub fn synthetic_image_pairs(
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
pub fn psnr_from_mse(mse: f64) -> f64 {
    if mse < 1e-12 {
        100.0 // Perfect reconstruction
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
