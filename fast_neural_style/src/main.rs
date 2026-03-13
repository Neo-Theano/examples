//! Fast Neural Style Transfer training example.
//!
//! Trains on synthetic data and saves the model to `style_model.safetensors`.

use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;
use fast_neural_style::{
    content_loss, style_loss, synthetic_image, FeatureExtractor, TransformerNet,
};

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

    // Save the trained model
    let sd = transformer.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("style_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to style_model.safetensors");
}
