//! ResNet-18 ImageNet model library.
//!
//! Provides the ResNet-18 architecture and helper functions for training
//! and inference. Supports serialization via state_dict.

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{
    BatchNorm1d, Conv2d, Linear, MaxPool2d, AdaptiveAvgPool2d, Flatten, ReLU, Module,
};

// ---------------------------------------------------------------------------
// BatchNorm2d -- normalise over (N, H, W) for each channel.
// The library only ships BatchNorm1d so we reshape [N,C,H,W] -> [N*H*W, C],
// apply BatchNorm1d, then reshape back.
// ---------------------------------------------------------------------------
pub struct BatchNorm2d {
    pub inner: BatchNorm1d,
    pub num_features: usize,
    pub weight: Variable,
    pub bias: Variable,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        let inner = BatchNorm1d::new(num_features);
        let params = inner.parameters();
        let weight = params[0].clone();
        let bias = params[1].clone();
        Self { inner, num_features, weight, bias }
    }

    pub fn from_state_dict(num_features: usize, weight: Tensor, bias: Tensor) -> Self {
        let w_var = Variable::requires_grad(weight);
        let b_var = Variable::requires_grad(bias);
        // Create a new BatchNorm1d then override its weight/bias by
        // constructing a fresh one. Since we cannot mutate inner's weight directly,
        // we store our own weight/bias and create a custom forward.
        let inner = BatchNorm1d::new(num_features);
        Self { inner, num_features, weight: w_var, bias: b_var }
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
pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    /// Optional 1x1 conv to match dimensions on the skip connection.
    pub downsample_conv: Option<Conv2d>,
    pub downsample_bn: Option<BatchNorm2d>,
}

impl ResidualBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
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

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        // conv1
        let c1_params = self.conv1.parameters();
        sd.insert(format!("{prefix}conv1.weight"), c1_params[0].tensor().clone());
        // bn1
        let bn1_params = self.bn1.parameters();
        sd.insert(format!("{prefix}bn1.weight"), bn1_params[0].tensor().clone());
        sd.insert(format!("{prefix}bn1.bias"), bn1_params[1].tensor().clone());
        // conv2
        let c2_params = self.conv2.parameters();
        sd.insert(format!("{prefix}conv2.weight"), c2_params[0].tensor().clone());
        // bn2
        let bn2_params = self.bn2.parameters();
        sd.insert(format!("{prefix}bn2.weight"), bn2_params[0].tensor().clone());
        sd.insert(format!("{prefix}bn2.bias"), bn2_params[1].tensor().clone());
        // downsample
        if let Some(ref dc) = self.downsample_conv {
            let dc_params = dc.parameters();
            sd.insert(format!("{prefix}downsample_conv.weight"), dc_params[0].tensor().clone());
        }
        if let Some(ref db) = self.downsample_bn {
            let db_params = db.parameters();
            sd.insert(format!("{prefix}downsample_bn.weight"), db_params[0].tensor().clone());
            sd.insert(format!("{prefix}downsample_bn.bias"), db_params[1].tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str, _in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2d::from_tensors(
            sd[&format!("{prefix}conv1.weight")].clone(),
            None,
            (stride, stride),
            (1, 1),
        );
        let bn1 = BatchNorm2d::from_state_dict(
            out_channels,
            sd[&format!("{prefix}bn1.weight")].clone(),
            sd[&format!("{prefix}bn1.bias")].clone(),
        );
        let conv2 = Conv2d::from_tensors(
            sd[&format!("{prefix}conv2.weight")].clone(),
            None,
            (1, 1),
            (1, 1),
        );
        let bn2 = BatchNorm2d::from_state_dict(
            out_channels,
            sd[&format!("{prefix}bn2.weight")].clone(),
            sd[&format!("{prefix}bn2.bias")].clone(),
        );

        let downsample_conv = sd.get(&format!("{prefix}downsample_conv.weight")).map(|w| {
            Conv2d::from_tensors(w.clone(), None, (stride, stride), (0, 0))
        });
        let downsample_bn = sd.get(&format!("{prefix}downsample_bn.weight")).map(|w| {
            BatchNorm2d::from_state_dict(
                out_channels,
                w.clone(),
                sd[&format!("{prefix}downsample_bn.bias")].clone(),
            )
        });

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
pub struct ResNet18 {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub relu: ReLU,
    pub maxpool: MaxPool2d,
    pub layer1: Vec<ResidualBlock>,
    pub layer2: Vec<ResidualBlock>,
    pub layer3: Vec<ResidualBlock>,
    pub layer4: Vec<ResidualBlock>,
    pub avgpool: AdaptiveAvgPool2d,
    pub flatten: Flatten,
    pub fc: Linear,
}

impl ResNet18 {
    pub fn new(num_classes: usize) -> Self {
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

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // conv1
        let c1_params = self.conv1.parameters();
        sd.insert("conv1.weight".to_string(), c1_params[0].tensor().clone());

        // bn1
        let bn1_params = self.bn1.parameters();
        sd.insert("bn1.weight".to_string(), bn1_params[0].tensor().clone());
        sd.insert("bn1.bias".to_string(), bn1_params[1].tensor().clone());

        // layer1-4
        let layers: [(&Vec<ResidualBlock>, &str); 4] = [
            (&self.layer1, "layer1"),
            (&self.layer2, "layer2"),
            (&self.layer3, "layer3"),
            (&self.layer4, "layer4"),
        ];
        for (layer, name) in &layers {
            for (i, block) in layer.iter().enumerate() {
                let prefix = format!("{name}.{i}.");
                sd.extend(block.state_dict(&prefix));
            }
        }

        // fc
        let fc_params = self.fc.parameters();
        sd.insert("fc.weight".to_string(), fc_params[0].tensor().clone());
        sd.insert("fc.bias".to_string(), fc_params[1].tensor().clone());

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, _num_classes: usize) -> Self {
        let conv1 = Conv2d::from_tensors(
            sd["conv1.weight"].clone(),
            None,
            (2, 2),
            (3, 3),
        );
        let bn1 = BatchNorm2d::from_state_dict(
            64,
            sd["bn1.weight"].clone(),
            sd["bn1.bias"].clone(),
        );

        // Helper to build a layer of residual blocks
        let build_layer = |name: &str, configs: &[(usize, usize, usize)]| -> Vec<ResidualBlock> {
            configs.iter().enumerate().map(|(i, &(inc, outc, s))| {
                let prefix = format!("{name}.{i}.");
                ResidualBlock::from_state_dict(sd, &prefix, inc, outc, s)
            }).collect()
        };

        let layer1 = build_layer("layer1", &[(64, 64, 1), (64, 64, 1)]);
        let layer2 = build_layer("layer2", &[(64, 128, 2), (128, 128, 1)]);
        let layer3 = build_layer("layer3", &[(128, 256, 2), (256, 256, 1)]);
        let layer4 = build_layer("layer4", &[(256, 512, 2), (512, 512, 1)]);

        let fc = Linear::from_tensors(
            sd["fc.weight"].clone(),
            Some(sd["fc.bias"].clone()),
        );

        Self {
            conv1,
            bn1,
            relu: ReLU,
            maxpool: MaxPool2d::new(3).with_stride((2, 2)).with_padding((1, 1)),
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool: AdaptiveAvgPool2d::new((1, 1)),
            flatten: Flatten::new(),
            fc,
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
pub fn random_tensor(shape: &[usize]) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Tensor::from_slice(&data, shape)
}

pub fn random_labels(batch_size: usize, num_classes: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..batch_size)
        .map(|_| rng.gen_range(0..num_classes) as f64)
        .collect();
    Tensor::from_slice(&data, &[batch_size])
}

pub fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
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
