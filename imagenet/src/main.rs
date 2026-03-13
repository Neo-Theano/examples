//! ResNet-18 ImageNet Training Example
//!
//! Demonstrates a simplified ResNet-18 architecture trained on synthetic
//! ImageNet-like data (224x224 images, 1000 classes).

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{
    BatchNorm1d, Conv2d, Linear, MaxPool2d, AdaptiveAvgPool2d, Flatten, ReLU, CrossEntropyLoss,
    Module,
};
use theano_optim::{Optimizer, SGD};

// ---------------------------------------------------------------------------
// BatchNorm2d — normalise over (N, H, W) for each channel.
// The library only ships BatchNorm1d so we reshape [N,C,H,W] -> [N*H*W, C],
// apply BatchNorm1d, then reshape back.
// ---------------------------------------------------------------------------
struct BatchNorm2d {
    inner: BatchNorm1d,
}

impl BatchNorm2d {
    fn new(num_features: usize) -> Self {
        Self {
            inner: BatchNorm1d::new(num_features),
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape().to_vec();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        // [N, C, H, W] -> [N*H*W, C]
        let data = input.tensor().to_vec_f64().unwrap();
        let nhw = n * h * w;
        let mut transposed = vec![0.0f64; nhw * c];
        for batch in 0..n {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let src = batch * c * h * w + ch * h * w + y * w + x;
                        let dst = (batch * h * w + y * w + x) * c + ch;
                        transposed[dst] = data[src];
                    }
                }
            }
        }
        let flat = Variable::new(Tensor::from_slice(&transposed, &[nhw, c]));
        let normed = self.inner.forward(&flat);
        // [N*H*W, C] -> [N, C, H, W]
        let normed_data = normed.tensor().to_vec_f64().unwrap();
        let mut output = vec![0.0f64; n * c * h * w];
        for batch in 0..n {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let src = (batch * h * w + y * w + x) * c + ch;
                        let dst = batch * c * h * w + ch * h * w + y * w + x;
                        output[dst] = normed_data[src];
                    }
                }
            }
        }
        Variable::new(Tensor::from_slice(&output, &[n, c, h, w]))
    }

    fn parameters(&self) -> Vec<Variable> {
        self.inner.parameters()
    }
}

// ---------------------------------------------------------------------------
// Residual Block
// ---------------------------------------------------------------------------
struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    /// Optional 1x1 conv to match dimensions on the skip connection.
    downsample_conv: Option<Conv2d>,
    downsample_bn: Option<BatchNorm2d>,
}

impl ResidualBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2d::with_options(
            in_channels, out_channels, (3, 3), (stride, stride), (1, 1), false,
        );
        let bn1 = BatchNorm2d::new(out_channels);
        let conv2 = Conv2d::with_options(
            out_channels, out_channels, (3, 3), (1, 1), (1, 1), false,
        );
        let bn2 = BatchNorm2d::new(out_channels);

        let (downsample_conv, downsample_bn) = if stride != 1 || in_channels != out_channels {
            (
                Some(Conv2d::with_options(
                    in_channels, out_channels, (1, 1), (stride, stride), (0, 0), false,
                )),
                Some(BatchNorm2d::new(out_channels)),
            )
        } else {
            (None, None)
        };

        Self { conv1, bn1, conv2, bn2, downsample_conv, downsample_bn }
    }
}

impl Module for ResidualBlock {
    fn forward(&self, input: &Variable) -> Variable {
        // Main path: conv -> bn -> relu -> conv -> bn
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(&out);
        let out = ReLU.forward(&out);
        let out = self.conv2.forward(&out);
        let out = self.bn2.forward(&out);

        // Skip connection (with optional downsample)
        let identity = if let (Some(ref dc), Some(ref db)) =
            (&self.downsample_conv, &self.downsample_bn)
        {
            let x = dc.forward(input);
            db.forward(&x)
        } else {
            input.clone()
        };

        // Residual add + ReLU
        let sum = out.add(&identity).unwrap();
        ReLU.forward(&sum)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref dc) = self.downsample_conv {
            params.extend(dc.parameters());
        }
        if let Some(ref db) = self.downsample_bn {
            params.extend(db.parameters());
        }
        params
    }
}

// ---------------------------------------------------------------------------
// ResNet-18
// ---------------------------------------------------------------------------
struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    layer4: Vec<ResidualBlock>,
    avgpool: AdaptiveAvgPool2d,
    flatten: Flatten,
    fc: Linear,
}

impl ResNet18 {
    fn new(num_classes: usize) -> Self {
        Self {
            conv1: Conv2d::with_options(3, 64, (7, 7), (2, 2), (3, 3), false),
            bn1: BatchNorm2d::new(64),
            relu: ReLU,
            maxpool: MaxPool2d::new(3).with_stride((2, 2)).with_padding((1, 1)),
            layer1: vec![
                ResidualBlock::new(64, 64, 1),
                ResidualBlock::new(64, 64, 1),
            ],
            layer2: vec![
                ResidualBlock::new(64, 128, 2),
                ResidualBlock::new(128, 128, 1),
            ],
            layer3: vec![
                ResidualBlock::new(128, 256, 2),
                ResidualBlock::new(256, 256, 1),
            ],
            layer4: vec![
                ResidualBlock::new(256, 512, 2),
                ResidualBlock::new(512, 512, 1),
            ],
            avgpool: AdaptiveAvgPool2d::new((1, 1)),
            flatten: Flatten::new(),
            fc: Linear::new(512, num_classes),
        }
    }
}

impl Module for ResNet18 {
    fn forward(&self, input: &Variable) -> Variable {
        let mut x = self.conv1.forward(input);
        x = self.bn1.forward(&x);
        x = self.relu.forward(&x);
        x = self.maxpool.forward(&x);

        for block in &self.layer1 { x = block.forward(&x); }
        for block in &self.layer2 { x = block.forward(&x); }
        for block in &self.layer3 { x = block.forward(&x); }
        for block in &self.layer4 { x = block.forward(&x); }

        x = self.avgpool.forward(&x);
        x = self.flatten.forward(&x);
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        for block in &self.layer1 { params.extend(block.parameters()); }
        for block in &self.layer2 { params.extend(block.parameters()); }
        for block in &self.layer3 { params.extend(block.parameters()); }
        for block in &self.layer4 { params.extend(block.parameters()); }
        params.extend(self.fc.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// Synthetic data helpers
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

fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
    let logits_data = logits.to_vec_f64().unwrap();
    let targets_data = targets.to_vec_f64().unwrap();
    let batch_size = targets.shape()[0];
    let num_classes = logits.shape()[1];

    let mut correct = 0;
    for i in 0..batch_size {
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;
        for c in 0..num_classes {
            let val = logits_data[i * num_classes + c];
            if val > max_val {
                max_val = val;
                max_idx = c;
            }
        }
        if max_idx == targets_data[i] as usize {
            correct += 1;
        }
    }
    correct as f64 / batch_size as f64
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    println!("=== ResNet-18 ImageNet Training (Synthetic Data) ===\n");

    let num_classes = 1000;
    // Use a tiny batch and spatial size for fast demonstration.
    let batch_size = 2;
    let image_h = 224;
    let image_w = 224;
    let num_epochs = 3;

    let model = ResNet18::new(num_classes);
    let num_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("Model parameters: {}", num_params);

    let criterion = CrossEntropyLoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01)
        .momentum(0.9)
        .weight_decay(1e-4);

    for epoch in 0..num_epochs {
        // Generate synthetic mini-batch
        let images = random_tensor(&[batch_size, 3, image_h, image_w]);
        let labels = random_labels(batch_size, num_classes);

        optimizer.zero_grad();

        let input = Variable::new(images);
        let target = Variable::new(labels.clone());

        let logits = model.forward(&input);
        let loss = criterion.forward(&logits, &target);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        let accuracy = compute_accuracy(logits.tensor(), &labels);

        println!(
            "Epoch [{}/{}]  Loss: {:.4}  Top-1 Accuracy: {:.2}%",
            epoch + 1,
            num_epochs,
            loss_val,
            accuracy * 100.0
        );
    }

    println!("\nTraining complete.");
}
