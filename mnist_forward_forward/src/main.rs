//! MNIST Forward-Forward Example — Hinton's Forward-Forward Algorithm (2022).
//!
//! Instead of backpropagation, each layer is trained independently with a local
//! "goodness" objective. Positive data (real images with correct labels) should
//! produce high goodness, while negative data (images with wrong labels) should
//! produce low goodness.
//!
//! Goodness = sum of squared activations in the layer.
//!
//! Each layer has its own optimizer and is trained to:
//!   - Increase goodness above a threshold for positive data
//!   - Decrease goodness below a threshold for negative data
//!
//! This is a fundamentally different training paradigm from backpropagation.

use rand::Rng;
use theano::prelude::*;
use theano_nn::{Linear, Module};
use theano_optim::{Adam, Optimizer};

/// A single Forward-Forward layer.
///
/// Learns a representation where positive examples have high "goodness"
/// (sum of squared activations) and negative examples have low goodness.
struct FFLayer {
    linear: Linear,
    threshold: f64,
}

impl FFLayer {
    fn new(in_features: usize, out_features: usize, threshold: f64) -> Self {
        Self {
            linear: Linear::new(in_features, out_features),
            threshold,
        }
    }

    /// Forward pass: Linear -> ReLU -> LayerNorm (simplified).
    /// Returns the activations.
    fn forward(&self, x: &Variable) -> Variable {
        let h = self.linear.forward(x);
        let h = h.relu().unwrap();

        // Simplified layer normalization: normalize each sample to unit norm
        // This prevents the goodness from growing unboundedly
        let h_data = h.tensor().to_vec_f64().unwrap();
        let shape = h.tensor().shape().to_vec();
        let batch = shape[0];
        let features = shape[1];

        let mut normed = vec![0.0f64; batch * features];
        for b in 0..batch {
            let mut norm_sq = 0.0f64;
            for j in 0..features {
                let v = h_data[b * features + j];
                norm_sq += v * v;
            }
            let norm = (norm_sq + 1e-8).sqrt();
            for j in 0..features {
                normed[b * features + j] = h_data[b * features + j] / norm;
            }
        }

        Variable::new(Tensor::from_slice(&normed, &shape))
    }

    /// Compute the "goodness" of the activations: mean of sum-of-squares per sample.
    ///
    /// goodness_i = sum_j(h_ij^2) for sample i
    /// Returns: [batch_size] tensor of goodness values
    fn goodness(activations: &Variable) -> Vec<f64> {
        let data = activations.tensor().to_vec_f64().unwrap();
        let shape = activations.tensor().shape().to_vec();
        let batch = shape[0];
        let features = shape[1];

        let mut good = vec![0.0f64; batch];
        for b in 0..batch {
            for j in 0..features {
                let v = data[b * features + j];
                good[b] += v * v;
            }
        }
        good
    }

    fn parameters(&self) -> Vec<Variable> {
        self.linear.parameters()
    }
}

/// A Forward-Forward network composed of multiple FF layers.
struct FFNetwork {
    layers: Vec<FFLayer>,
}

impl FFNetwork {
    fn new(layer_sizes: &[usize], threshold: f64) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(FFLayer::new(layer_sizes[i], layer_sizes[i + 1], threshold));
        }
        Self { layers }
    }

    /// Create positive data by overlaying the label into the first 10 pixels of the image.
    ///
    /// For positive data: use the correct label.
    /// For negative data: use a random incorrect label.
    fn overlay_label(image: &[f64], label: usize, num_classes: usize) -> Vec<f64> {
        let mut data = image.to_vec();
        // Zero out the first num_classes positions, then set the label position to 1
        for i in 0..num_classes.min(data.len()) {
            data[i] = 0.0;
        }
        if label < data.len() {
            data[label] = 1.0;
        }
        data
    }

    /// Train the network using the Forward-Forward algorithm.
    fn train_epoch(
        &self,
        optimizers: &mut [Adam],
        images: &[Vec<f64>],
        labels: &[usize],
        num_classes: usize,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let batch_size = images.len();
        let input_dim = images[0].len();
        let mut total_loss = 0.0;

        // Create positive samples: images with correct labels
        let mut pos_data = Vec::with_capacity(batch_size * input_dim);
        for i in 0..batch_size {
            let overlaid = Self::overlay_label(&images[i], labels[i], num_classes);
            pos_data.extend_from_slice(&overlaid);
        }

        // Create negative samples: images with random incorrect labels
        let mut neg_data = Vec::with_capacity(batch_size * input_dim);
        for i in 0..batch_size {
            let mut wrong_label = rng.gen_range(0..num_classes);
            while wrong_label == labels[i] {
                wrong_label = rng.gen_range(0..num_classes);
            }
            let overlaid = Self::overlay_label(&images[i], wrong_label, num_classes);
            neg_data.extend_from_slice(&overlaid);
        }

        let mut pos_input =
            Variable::new(Tensor::from_slice(&pos_data, &[batch_size, input_dim]));
        let mut neg_input =
            Variable::new(Tensor::from_slice(&neg_data, &[batch_size, input_dim]));

        // Train each layer independently
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            optimizers[layer_idx].zero_grad();

            // Forward pass for positive and negative data
            let pos_act = layer.forward(&pos_input);
            let neg_act = layer.forward(&neg_input);

            // Compute goodness
            let pos_good = FFLayer::goodness(&pos_act);
            let neg_good = FFLayer::goodness(&neg_act);

            // Forward-Forward loss:
            // For positive data: loss = -log(sigmoid(goodness - threshold))
            // For negative data: loss = -log(sigmoid(threshold - goodness))
            // Simplified: we want pos_good > threshold and neg_good < threshold
            let threshold = layer.threshold;
            let mut layer_loss = 0.0f64;

            for b in 0..batch_size {
                // Positive: push goodness above threshold
                let pos_margin = pos_good[b] - threshold;
                layer_loss += softplus(-pos_margin);

                // Negative: push goodness below threshold
                let neg_margin = threshold - neg_good[b];
                layer_loss += softplus(-neg_margin);
            }
            layer_loss /= (2 * batch_size) as f64;

            // Since the FF algorithm trains each layer independently,
            // we create a differentiable loss using Variable ops.
            // Compute loss as Variable for autograd:
            let pos_act_rg = {
                let p = layer.linear.forward(&pos_input);
                p.relu().unwrap()
            };
            let neg_act_rg = {
                let p = layer.linear.forward(&neg_input);
                p.relu().unwrap()
            };

            // Goodness = sum of squares per sample
            let pos_sq = pos_act_rg.mul(&pos_act_rg).unwrap();
            let neg_sq = neg_act_rg.mul(&neg_act_rg).unwrap();

            // Sum over features for each sample, then mean over batch
            let pos_goodness = pos_sq.sum().unwrap().mul_scalar(1.0 / batch_size as f64).unwrap();
            let neg_goodness = neg_sq.sum().unwrap().mul_scalar(1.0 / batch_size as f64).unwrap();

            // Loss: maximize positive goodness, minimize negative goodness
            // loss = -pos_goodness + neg_goodness (gradient pushes the right way)
            let loss = neg_goodness.sub(&pos_goodness).unwrap();

            loss.backward();
            optimizers[layer_idx].step();

            total_loss += layer_loss;

            // Pass activations to the next layer (detached from graph)
            pos_input = pos_act.detach();
            neg_input = neg_act.detach();
        }

        total_loss / self.layers.len() as f64
    }

    /// Predict labels using the trained network.
    ///
    /// For each possible label, overlay it on the image, run through the network,
    /// and pick the label that produces the highest total goodness.
    fn predict(&self, images: &[Vec<f64>], num_classes: usize) -> Vec<usize> {
        let batch_size = images.len();
        let input_dim = images[0].len();

        let mut predictions = vec![0usize; batch_size];

        for b in 0..batch_size {
            let mut best_label = 0;
            let mut best_goodness = f64::NEG_INFINITY;

            for label in 0..num_classes {
                let overlaid = Self::overlay_label(&images[b], label, num_classes);
                let mut x = Variable::new(Tensor::from_slice(&overlaid, &[1, input_dim]));

                let mut total_goodness = 0.0;
                for layer in &self.layers {
                    let act = layer.forward(&x);
                    let good = FFLayer::goodness(&act);
                    total_goodness += good[0];
                    x = act.detach();
                }

                if total_goodness > best_goodness {
                    best_goodness = total_goodness;
                    best_label = label;
                }
            }

            predictions[b] = best_label;
        }

        predictions
    }
}

/// Numerically stable softplus: log(1 + exp(x)).
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Print the model summary.
fn print_model_summary(network: &FFNetwork) {
    println!("=== Forward-Forward Network Architecture ===");
    for (i, layer) in network.layers.iter().enumerate() {
        let params: usize = layer.parameters().iter().map(|p| p.tensor().numel()).sum();
        let w_shape = layer.linear.weight().tensor().shape().to_vec();
        println!(
            "  Layer {}: Linear({}, {}) -> ReLU -> Normalize  [{} params, threshold={:.1}]",
            i,
            w_shape[1],
            w_shape[0],
            params,
            layer.threshold
        );
    }
    let total: usize = network
        .layers
        .iter()
        .flat_map(|l| l.parameters())
        .map(|p| p.tensor().numel())
        .sum();
    println!("=============================================");
    println!("Total trainable parameters: {}", total);
    println!();
}

fn main() {
    println!("Neo Theano — MNIST Forward-Forward Algorithm Example");
    println!("(Hinton 2022: Training without backpropagation)\n");

    // Hyperparameters
    let input_dim = 784; // 28*28 flattened
    let num_classes = 10;
    let threshold = 2.0;
    let lr = 0.001;
    let num_epochs = 5;
    let batch_size = 8;
    let train_batches = 10;
    let test_batches = 5;

    // Build network: 784 -> 500 -> 500 -> 500
    let layer_sizes = vec![input_dim, 500, 500, 500];
    let network = FFNetwork::new(&layer_sizes, threshold);
    print_model_summary(&network);

    // Create per-layer optimizers (each layer trained independently!)
    let mut optimizers: Vec<Adam> = network
        .layers
        .iter()
        .map(|layer| Adam::new(layer.parameters(), lr))
        .collect();

    // Training loop
    println!("Training with Forward-Forward algorithm...");
    println!("(Each layer trained independently - no backprop through the network!)\n");

    let mut rng = rand::thread_rng();

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..train_batches {
            // Generate synthetic batch
            let images: Vec<Vec<f64>> = (0..batch_size)
                .map(|_| (0..input_dim).map(|_| rng.gen::<f64>()).collect())
                .collect();
            let labels: Vec<usize> = (0..batch_size)
                .map(|_| rng.gen_range(0..num_classes))
                .collect();

            let loss = network.train_epoch(&mut optimizers, &images, &labels, num_classes);
            epoch_loss += loss;

            if batch_idx % 5 == 0 {
                println!(
                    "  Epoch [{}/{}], Batch [{}/{}], Loss: {:.4}",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    train_batches,
                    loss
                );
            }
        }

        let avg_loss = epoch_loss / train_batches as f64;
        println!(
            "  Epoch [{}/{}] Average Loss: {:.4}",
            epoch + 1,
            num_epochs,
            avg_loss
        );
    }

    // Evaluation
    println!("\nEvaluating...");
    let mut total_correct = 0;
    let mut total_samples = 0;

    for _ in 0..test_batches {
        let images: Vec<Vec<f64>> = (0..batch_size)
            .map(|_| (0..input_dim).map(|_| rng.gen::<f64>()).collect())
            .collect();
        let labels: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..num_classes))
            .collect();

        let predictions = network.predict(&images, num_classes);

        for i in 0..batch_size {
            if predictions[i] == labels[i] {
                total_correct += 1;
            }
        }
        total_samples += batch_size;
    }

    let test_acc = total_correct as f64 / total_samples as f64;
    println!(
        "Test Accuracy: {:.2}% ({}/{} correct)",
        test_acc * 100.0,
        total_correct,
        total_samples
    );
    println!("(Note: accuracy is ~10% random baseline since we use synthetic data)");
    println!("\nKey insight: No gradient flows between layers!");
    println!("Each layer is trained with its own local objective (goodness).");
}
