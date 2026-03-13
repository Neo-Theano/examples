//! Vision Transformer (ViT) Example
//!
//! Implements a Vision Transformer for image classification on synthetic
//! CIFAR-like data (32x32 images, 10 classes).
//! Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021).

use theano_autograd::Variable;
use theano_nn::{CrossEntropyLoss, Module};
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use vision_transformer::{ViT, random_images, random_labels};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    println!("=== Vision Transformer (ViT) Training (Synthetic Data) ===\n");

    let img_channels = 3;
    let img_size = 32;
    let patch_size = 4;
    let embed_dim = 64;
    let num_heads = 4;
    let num_blocks = 4;
    let num_classes = 10;
    let batch_size = 4;
    let num_epochs = 5;
    let lr = 0.001;

    let num_patches = (img_size / patch_size) * (img_size / patch_size);

    let model = ViT::new(
        img_channels,
        img_size,
        patch_size,
        embed_dim,
        num_heads,
        num_blocks,
        num_classes,
    );

    let total_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("ViT Configuration:");
    println!("  Image size:   {}x{}", img_size, img_size);
    println!("  Patch size:   {}x{}", patch_size, patch_size);
    println!("  Num patches:  {}", num_patches);
    println!("  Embed dim:    {}", embed_dim);
    println!("  Num heads:    {}", num_heads);
    println!("  Num blocks:   {}", num_blocks);
    println!("  Num classes:  {}", num_classes);
    println!("  Total params: {}\n", total_params);

    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), lr);

    for epoch in 0..num_epochs {
        let images = random_images(batch_size, img_channels, img_size, img_size);
        let labels = random_labels(batch_size, num_classes);

        optimizer.zero_grad();

        let input = Variable::new(images);
        let target = Variable::new(labels);

        let logits = model.forward(&input);
        let loss = criterion.forward(&logits, &target);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        println!(
            "Epoch [{}/{}]  Loss: {:.4}",
            epoch + 1,
            num_epochs,
            loss_val,
        );
    }

    // Save model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("vit_model.safetensors", &bytes).expect("failed to save model");
    println!("\nModel saved to vit_model.safetensors ({} bytes)", bytes.len());

    println!("\nTraining complete.");
}
