//! ResNet-18 ImageNet Training Example
//!
//! Demonstrates a simplified ResNet-18 architecture trained on synthetic
//! ImageNet-like data (224x224 images, 1000 classes).

use theano_autograd::Variable;
use theano_nn::{CrossEntropyLoss, Module};
use theano_optim::{Optimizer, SGD};
use theano_serialize::save_state_dict;

use imagenet::{ResNet18, random_tensor, random_labels, compute_accuracy};

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

    // Save model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("resnet18_model.safetensors", &bytes).expect("failed to save model");
    println!("\nModel saved to resnet18_model.safetensors ({} bytes)", bytes.len());

    println!("\nTraining complete.");
}
