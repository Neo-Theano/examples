//! Fast Neural Style Transfer example.
//!
//! Implements a simplified style transfer network:
//! - Encoder: Conv2d layers with downsampling
//! - Residual blocks: Conv2d -> ReLU -> Conv2d + skip connection
//! - Decoder: Conv2d layers (simulating upsampling)
//! - Content loss: MSE between feature maps
//! - Style loss: MSE between Gram matrices
//!
//! Demonstrates the architecture and a forward pass with synthetic data.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Conv2d, Module, MSELoss};
use theano_optim::{Adam, Optimizer};

// ---------------------------------------------------------------------------
// Residual Block
// ---------------------------------------------------------------------------

struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualBlock {
    fn new(channels: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
            conv2: Conv2d::with_options(channels, channels, (3, 3), (1, 1), (1, 1), true),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let residual = x.clone();
        let out = self.conv1.forward(x);
        let out = out.relu().unwrap();
        let out = self.conv2.forward(&out);
        // Skip connection: out + residual
        out.add(&residual).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Style Transfer Network (Transformer Network)
// ---------------------------------------------------------------------------

struct TransformerNet {
    // Encoder (downsampling)
    enc_conv1: Conv2d,   // 3 -> 32, stride 1
    enc_conv2: Conv2d,   // 32 -> 64, stride 2
    enc_conv3: Conv2d,   // 64 -> 128, stride 2

    // Residual blocks
    res1: ResidualBlock,
    res2: ResidualBlock,

    // Decoder (upsampling via conv — simplified, no actual upsample)
    dec_conv1: Conv2d,   // 128 -> 64
    dec_conv2: Conv2d,   // 64 -> 32
    dec_conv3: Conv2d,   // 32 -> 3
}

impl TransformerNet {
    fn new() -> Self {
        Self {
            // Encoder
            enc_conv1: Conv2d::with_options(3, 32, (3, 3), (1, 1), (1, 1), true),
            enc_conv2: Conv2d::with_options(32, 64, (3, 3), (2, 2), (1, 1), true),
            enc_conv3: Conv2d::with_options(64, 128, (3, 3), (2, 2), (1, 1), true),

            // Residual blocks
            res1: ResidualBlock::new(128),
            res2: ResidualBlock::new(128),

            // Decoder (keeps spatial dims same — simplified)
            dec_conv1: Conv2d::with_options(128, 64, (3, 3), (1, 1), (1, 1), true),
            dec_conv2: Conv2d::with_options(64, 32, (3, 3), (1, 1), (1, 1), true),
            dec_conv3: Conv2d::with_options(32, 3, (3, 3), (1, 1), (1, 1), true),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
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

    fn parameters(&self) -> Vec<Variable> {
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
}

// ---------------------------------------------------------------------------
// Feature Extractor (simplified VGG-like)
// ---------------------------------------------------------------------------

struct FeatureExtractor {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl FeatureExtractor {
    fn new() -> Self {
        Self {
            conv1: Conv2d::with_options(3, 16, (3, 3), (1, 1), (1, 1), true),
            conv2: Conv2d::with_options(16, 32, (3, 3), (1, 1), (1, 1), true),
        }
    }

    /// Extract feature maps at two levels.
    fn forward(&self, x: &Variable) -> (Variable, Variable) {
        let feat1 = self.conv1.forward(x).relu().unwrap();
        let feat2 = self.conv2.forward(&feat1).relu().unwrap();
        (feat1, feat2)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Loss Functions
// ---------------------------------------------------------------------------

/// Compute Gram matrix for style loss. Input: [N, C, H, W] -> [N, C, C]
fn gram_matrix(features: &Variable) -> Variable {
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
fn content_loss(features_output: &Variable, features_target: &Variable) -> Variable {
    let mse = MSELoss::new();
    mse.forward(features_output, features_target)
}

/// Style loss: MSE between Gram matrices.
fn style_loss(features_output: &Variable, features_style: &Variable) -> Variable {
    let gram_out = gram_matrix(features_output);
    let gram_style = gram_matrix(features_style);
    let mse = MSELoss::new();
    mse.forward(&gram_out, &gram_style)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic image [batch, 3, H, W] with values in [0, 1].
fn synthetic_image(batch_size: usize, h: usize, w: usize) -> Variable {
    let mut rng = rand::thread_rng();
    let numel = batch_size * 3 * h * w;
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Variable::new(Tensor::from_slice(&data, &[batch_size, 3, h, w]))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Fast Neural Style Transfer ===");
    println!();

    let img_h = 16;
    let img_w = 16;
    let batch_size = 2;
    let num_epochs = 15;
    let batches_per_epoch = 5;
    let content_weight = 1.0;
    let style_weight = 1e5;

    let transformer = TransformerNet::new();
    let feature_extractor = FeatureExtractor::new();

    let transformer_params = transformer.parameters();
    let param_count: usize = transformer_params.iter().map(|p| p.tensor().numel()).sum();
    let feat_params: usize = feature_extractor.parameters().iter().map(|p| p.tensor().numel()).sum();

    println!("Image size: {}x{} (3 channels)", img_h, img_w);
    println!("Transformer parameters: {}", param_count);
    println!("Feature extractor parameters: {}", feat_params);
    println!("Batch size: {}", batch_size);
    println!("Content weight: {}", content_weight);
    println!("Style weight: {}", style_weight);
    println!("Epochs: {}", num_epochs);
    println!();

    let mut optimizer = Adam::new(transformer_params, 1e-3);

    // Generate a fixed "style image" for reference
    let style_image = synthetic_image(batch_size, img_h, img_w);
    let (_style_feat1, _style_feat2) = feature_extractor.forward(&style_image);

    // Compute the decoder output spatial size
    // enc_conv1: same, enc_conv2: stride 2 -> /2, enc_conv3: stride 2 -> /2
    // Then decoder keeps same spatial dims (no upsample)
    let enc_h = ((img_h + 2 - 3) / 2 + 1 + 2 - 3) / 2 + 1; // after two stride-2 convs
    let enc_w = ((img_w + 2 - 3) / 2 + 1 + 2 - 3) / 2 + 1;
    println!("Encoder output spatial: {}x{}", enc_h, enc_w);
    println!("(Decoder output will match encoder's downsampled size)");
    println!();

    for epoch in 1..=num_epochs {
        let mut total_content = 0.0;
        let mut total_style = 0.0;
        let mut total_loss = 0.0;

        for _ in 0..batches_per_epoch {
            optimizer.zero_grad();

            // Content image (what we want to stylize)
            let content_image = synthetic_image(batch_size, img_h, img_w);

            // Run through transformer
            let stylized = transformer.forward(&content_image);

            // Extract features from the encoder output (decoder output is smaller)
            // For content loss, compare features of the content input vs stylized output
            // Since transformer changes spatial size, we extract features from same-size data
            let stylized_shape = stylized.tensor().shape().to_vec();
            let s_h = stylized_shape[2];
            let s_w = stylized_shape[3];

            // Generate content target at the same spatial resolution as decoder output
            let content_target = synthetic_image(batch_size, s_h, s_w);
            let style_target = synthetic_image(batch_size, s_h, s_w);

            let (out_feat1, out_feat2) = feature_extractor.forward(&stylized);
            let (_ct_feat1, ct_feat2) = feature_extractor.forward(&content_target);
            let (st_feat1, _st_feat2) = feature_extractor.forward(&style_target);

            // Content loss (from deeper features)
            let c_loss = content_loss(&out_feat2, &ct_feat2);
            let c_loss_val = c_loss.tensor().item().unwrap();

            // Style loss (from shallower features, using Gram matrices)
            let s_loss = style_loss(&out_feat1, &st_feat1);
            let s_loss_val = s_loss.tensor().item().unwrap();

            total_content += c_loss_val;
            total_style += s_loss_val;

            // Weighted total loss
            let weighted_content = c_loss.mul_scalar(content_weight).unwrap();
            let weighted_style = s_loss.mul_scalar(style_weight).unwrap();
            let loss = weighted_content.add(&weighted_style).unwrap();
            let loss_val = loss.tensor().item().unwrap();
            total_loss += loss_val;

            loss.backward();
            optimizer.step();
        }

        let avg_content = total_content / batches_per_epoch as f64;
        let avg_style = total_style / batches_per_epoch as f64;
        let avg_total = total_loss / batches_per_epoch as f64;

        println!(
            "Epoch [{:2}/{}]  Total: {:.4}  Content: {:.6}  Style: {:.6}",
            epoch, num_epochs, avg_total, avg_content, avg_style
        );
    }

    // Forward pass demo
    println!();
    println!("Running single forward pass...");
    let test_input = synthetic_image(1, img_h, img_w);
    let test_output = transformer.forward(&test_input);
    println!("Input shape:  {:?}", test_input.tensor().shape());
    println!("Output shape: {:?}", test_output.tensor().shape());

    let out_data = test_output.tensor().to_vec_f64().unwrap();
    let out_min = out_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let out_max = out_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let out_mean = out_data.iter().sum::<f64>() / out_data.len() as f64;
    println!(
        "Output stats: min={:.4}, max={:.4}, mean={:.4}",
        out_min, out_max, out_mean
    );
    println!();
    println!("Training complete.");
}
