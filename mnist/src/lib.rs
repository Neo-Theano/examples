//! MNIST CNN model definitions.
//!
//! Architecture mirrors the PyTorch MNIST example:
//!   Conv2d(1,32,3) -> ReLU -> Conv2d(32,64,3) -> ReLU -> MaxPool2d(2)
//!   -> Dropout(0.25) -> Flatten -> Linear(9216,128) -> ReLU
//!   -> Dropout(0.5) -> Linear(128,10)

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU};

/// CNN model for MNIST digit classification.
pub struct MnistCNN {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub pool: MaxPool2d,
    pub dropout1: Dropout,
    pub flatten: Flatten,
    pub fc1: Linear,
    pub relu: ReLU,
    pub dropout2: Dropout,
    pub fc2: Linear,
}

impl MnistCNN {
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            pool: MaxPool2d::new(2),
            dropout1: Dropout::new(0.25),
            flatten: Flatten::new(),
            fc1: Linear::new(9216, 128),
            relu: ReLU,
            dropout2: Dropout::new(0.5),
            fc2: Linear::new(128, 10),
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // Conv block 1: Conv2d(1,32,3) -> ReLU
        let x = self.conv1.forward(x);
        let x = x.relu().unwrap();

        // Conv block 2: Conv2d(32,64,3) -> ReLU -> MaxPool2d(2)
        let x = self.conv2.forward(&x);
        let x = x.relu().unwrap();
        let x = self.pool.forward(&x);

        // Dropout -> Flatten
        let x = self.dropout1.forward(&x);
        let x = self.flatten.forward(&x);

        // FC block: Linear(9216,128) -> ReLU -> Dropout(0.5) -> Linear(128,10)
        let x = self.fc1.forward(&x);
        let x = self.relu.forward(&x);
        let x = self.dropout2.forward(&x);
        self.fc2.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    pub fn set_eval(&mut self) {
        self.dropout1.eval();
        self.dropout2.eval();
    }

    pub fn set_train(&mut self) {
        self.dropout1.train();
        self.dropout2.train();
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // Conv1 parameters
        let conv1_params = self.conv1.parameters();
        sd.insert("conv1.weight".to_string(), conv1_params[0].tensor().clone());
        sd.insert("conv1.bias".to_string(), conv1_params[1].tensor().clone());

        // Conv2 parameters
        let conv2_params = self.conv2.parameters();
        sd.insert("conv2.weight".to_string(), conv2_params[0].tensor().clone());
        sd.insert("conv2.bias".to_string(), conv2_params[1].tensor().clone());

        // FC1 parameters
        for (name, param) in self.fc1.named_parameters() {
            sd.insert(format!("fc1.{name}"), param.tensor().clone());
        }

        // FC2 parameters
        for (name, param) in self.fc2.named_parameters() {
            sd.insert(format!("fc2.{name}"), param.tensor().clone());
        }

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            conv1: Conv2d::from_tensors(
                sd["conv1.weight"].clone(),
                Some(sd["conv1.bias"].clone()),
                (1, 1),
                (0, 0),
            ),
            conv2: Conv2d::from_tensors(
                sd["conv2.weight"].clone(),
                Some(sd["conv2.bias"].clone()),
                (1, 1),
                (0, 0),
            ),
            pool: MaxPool2d::new(2),
            dropout1: Dropout::new(0.25),
            flatten: Flatten::new(),
            fc1: Linear::from_tensors(
                sd["fc1.weight"].clone(),
                Some(sd["fc1.bias"].clone()),
            ),
            relu: ReLU,
            dropout2: Dropout::new(0.5),
            fc2: Linear::from_tensors(
                sd["fc2.weight"].clone(),
                Some(sd["fc2.bias"].clone()),
            ),
        }
    }
}

/// Print the model architecture and total parameter count.
pub fn print_model_summary(model: &MnistCNN) {
    println!("=== MNIST CNN Architecture ===");
    println!("  Conv2d(1, 32, kernel_size=3)    -> ReLU");
    println!("  Conv2d(32, 64, kernel_size=3)   -> ReLU -> MaxPool2d(2)");
    println!("  Dropout(0.25)");
    println!("  Flatten");
    println!("  Linear(9216, 128)               -> ReLU");
    println!("  Dropout(0.5)");
    println!("  Linear(128, 10)");
    println!("==============================");

    let total_params: usize = model
        .parameters()
        .iter()
        .map(|p| p.tensor().numel())
        .sum();
    println!("Total trainable parameters: {}", total_params);
    println!();
}

/// Generate a batch of synthetic MNIST-like data.
/// Returns (images: [N, 1, 28, 28], labels: [N]).
pub fn generate_batch(batch_size: usize) -> (Tensor, Tensor) {
    let mut rng = rand::thread_rng();
    let img_numel = batch_size * 1 * 28 * 28;
    let img_data: Vec<f64> = (0..img_numel).map(|_| rng.gen::<f64>()).collect();
    let label_data: Vec<f64> = (0..batch_size)
        .map(|_| rng.gen_range(0..10) as f64)
        .collect();

    let images = Tensor::from_slice(&img_data, &[batch_size, 1, 28, 28]);
    let labels = Tensor::from_slice(&label_data, &[batch_size]);
    (images, labels)
}

/// Compute accuracy: fraction of correct predictions.
pub fn accuracy(logits: &Tensor, labels: &Tensor) -> f64 {
    let n = logits.shape()[0];
    let c = logits.shape()[1];
    let logits_data = logits.to_vec_f64().unwrap();
    let labels_data = labels.to_vec_f64().unwrap();

    let mut correct = 0;
    for i in 0..n {
        let mut best_class = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..c {
            let v = logits_data[i * c + j];
            if v > best_val {
                best_val = v;
                best_class = j;
            }
        }
        if best_class == labels_data[i] as usize {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}
