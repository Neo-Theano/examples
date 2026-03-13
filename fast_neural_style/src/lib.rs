//! Fast Neural Style Transfer model definitions.
//!
//! Implements a simplified style transfer network:
//! - Encoder: Conv2d layers with downsampling
//! - Residual blocks: Conv2d -> ReLU -> Conv2d + skip connection
//! - Decoder: Conv2d layers (simulating upsampling)
//! - Content loss: MSE between feature maps
//! - Style loss: MSE between Gram matrices

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Conv2d, Module, MSELoss};

// ---------------------------------------------------------------------------
// Residual Block
// ---------------------------------------------------------------------------

pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl ResidualBlock {
    pub fn new(channels: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            conv2: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        let residual = x.clone();
        let out = self.conv1.forward(x);
        let out = out.relu().unwrap();
        let out = self.conv2.forward(&out);
        // Skip connection: out + residual
        out.add(&residual).unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.conv1.named_parameters() {
            sd.insert(format!("{prefix}conv1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.conv2.named_parameters() {
            sd.insert(format!("{prefix}conv2.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str, channels: usize) -> Self {
        let _ = channels; // stride and padding are fixed for residual blocks
        Self {
            conv1: Conv2d::from_tensors(
                sd[&format!("{prefix}conv1.weight")].clone(),
                Some(sd[&format!("{prefix}conv1.bias")].clone()),
                (1, 1),
                (1, 1),
            ),
            conv2: Conv2d::from_tensors(
                sd[&format!("{prefix}conv2.weight")].clone(),
                Some(sd[&format!("{prefix}conv2.bias")].clone()),
                (1, 1),
                (1, 1),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Style Transfer Network (Transformer Network)
// ---------------------------------------------------------------------------

pub struct TransformerNet {
    // Encoder (downsampling)
    pub enc_conv1: Conv2d,   // 3 -> 32, stride 1
    pub enc_conv2: Conv2d,   // 32 -> 64, stride 2
    pub enc_conv3: Conv2d,   // 64 -> 128, stride 2

    // Residual blocks
    pub res1: ResidualBlock,
    pub res2: ResidualBlock,

    // Decoder (upsampling via conv -- simplified, no actual upsample)
    pub dec_conv1: Conv2d,   // 128 -> 64
    pub dec_conv2: Conv2d,   // 64 -> 32
    pub dec_conv3: Conv2d,   // 32 -> 3
}

impl TransformerNet {
    pub fn new() -> Self {
        Self {
            // Encoder
            enc_conv1: Conv2d::with_options(3, 32, (3, 3), (1, 1), (1, 1), true),
            enc_conv2: Conv2d::with_options(32, 64, (3, 3), (2, 2), (1, 1), true),
            enc_conv3: Conv2d::with_options(64, 128, (3, 3), (2, 2), (1, 1), true),

            // Residual blocks
            res1: ResidualBlock::new(128),
            res2: ResidualBlock::new(128),

            // Decoder (keeps spatial dims same -- simplified)
            dec_conv1: Conv2d::with_options(128, 64, (3, 3), (1, 1), (1, 1), true),
            dec_conv2: Conv2d::with_options(64, 32, (3, 3), (1, 1), (1, 1), true),
            dec_conv3: Conv2d::with_options(32, 3, (3, 3), (1, 1), (1, 1), true),
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // Encoder
        let h = self.enc_conv1.forward(x).relu().unwrap();
        let h = self.enc_conv2.forward(&h).relu().unwrap();
        let h = self.enc_conv3.forward(&h).relu().unwrap();

        // Residual blocks
        let h = self.res1.forward(&h);
        let h = self.res2.forward(&h);

        // Decoder
        let h = self.dec_conv1.forward(&h).relu().unwrap();
        let h = self.dec_conv2.forward(&h).relu().unwrap();
        self.dec_conv3.forward(&h)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.enc_conv1.parameters();
        params.extend(self.enc_conv2.parameters());
        params.extend(self.enc_conv3.parameters());
        params.extend(self.res1.parameters());
        params.extend(self.res2.parameters());
        params.extend(self.dec_conv1.parameters());
        params.extend(self.dec_conv2.parameters());
        params.extend(self.dec_conv3.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.enc_conv1.named_parameters() {
            sd.insert(format!("enc_conv1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.enc_conv2.named_parameters() {
            sd.insert(format!("enc_conv2.{name}"), param.tensor().clone());
        }
        for (name, param) in self.enc_conv3.named_parameters() {
            sd.insert(format!("enc_conv3.{name}"), param.tensor().clone());
        }
        sd.extend(self.res1.state_dict("res_blocks.0."));
        sd.extend(self.res2.state_dict("res_blocks.1."));
        for (name, param) in self.dec_conv1.named_parameters() {
            sd.insert(format!("dec_conv1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.dec_conv2.named_parameters() {
            sd.insert(format!("dec_conv2.{name}"), param.tensor().clone());
        }
        for (name, param) in self.dec_conv3.named_parameters() {
            sd.insert(format!("dec_conv3.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            enc_conv1: Conv2d::from_tensors(
                sd["enc_conv1.weight"].clone(),
                Some(sd["enc_conv1.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
            enc_conv2: Conv2d::from_tensors(
                sd["enc_conv2.weight"].clone(),
                Some(sd["enc_conv2.bias"].clone()),
                (2, 2),
                (1, 1),
            ),
            enc_conv3: Conv2d::from_tensors(
                sd["enc_conv3.weight"].clone(),
                Some(sd["enc_conv3.bias"].clone()),
                (2, 2),
                (1, 1),
            ),
            res1: ResidualBlock::from_state_dict(sd, "res_blocks.0.", 128),
            res2: ResidualBlock::from_state_dict(sd, "res_blocks.1.", 128),
            dec_conv1: Conv2d::from_tensors(
                sd["dec_conv1.weight"].clone(),
                Some(sd["dec_conv1.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
            dec_conv2: Conv2d::from_tensors(
                sd["dec_conv2.weight"].clone(),
                Some(sd["dec_conv2.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
            dec_conv3: Conv2d::from_tensors(
                sd["dec_conv3.weight"].clone(),
                Some(sd["dec_conv3.bias"].clone()),
                (1, 1),
                (1, 1),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Feature Extractor (simplified VGG-like)
// ---------------------------------------------------------------------------

pub struct FeatureExtractor {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::with_options(3, 16, (3, 3), (1, 1), (1, 1), true),
            conv2: Conv2d::with_options(16, 32, (3, 3), (1, 1), (1, 1), true),
        }
    }

    /// Extract feature maps at two levels.
    pub fn forward(&self, x: &Variable) -> (Variable, Variable) {
        let feat1 = self.conv1.forward(x).relu().unwrap();
        let feat2 = self.conv2.forward(&feat1).relu().unwrap();
        (feat1, feat2)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------

/// Compute Gram matrix for style loss. Input: [N, C, H, W] -> [N, C, C]
pub fn gram_matrix(features: &Variable) -> Variable {
    let shape = features.tensor().shape();
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let spatial = h * w;

    let data = features.tensor().to_vec_f64().unwrap();
    let mut gram_data = vec![0.0f64; n * c * c];

    for batch in 0..n {
        for i in 0..c {
            for j in 0..c {
                let mut dot = 0.0;
                for k in 0..spatial {
                    let fi = data[batch * c * spatial + i * spatial + k];
                    let fj = data[batch * c * spatial + j * spatial + k];
                    dot += fi * fj;
                }
                gram_data[batch * c * c + i * c + j] = dot / spatial as f64;
            }
        }
    }

    Variable::new(Tensor::from_slice(&gram_data, &[n, c, c]))
}

/// Content loss: MSE between feature maps.
pub fn content_loss(features_output: &Variable, features_target: &Variable) -> Variable {
    let mse = MSELoss::new();
    mse.forward(features_output, features_target)
}

/// Style loss: MSE between Gram matrices.
pub fn style_loss(features_output: &Variable, features_style: &Variable) -> Variable {
    let gram_out = gram_matrix(features_output);
    let gram_style = gram_matrix(features_style);
    let mse = MSELoss::new();
    mse.forward(&gram_out, &gram_style)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic image [batch, 3, H, W] with values in [0, 1].
pub fn synthetic_image(batch_size: usize, h: usize, w: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let numel = batch_size * 3 * h * w;
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 3, h, w]))
}
