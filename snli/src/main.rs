//! SNLI — Natural Language Inference example.
//!
//! Trains on synthetic data and saves the model to `snli_model.safetensors`.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use snli::{
    SNLIClassifier, generate_nli_batch,
    VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES, SEQ_LEN, BATCH_SIZE,
};

const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.001;

fn main() {
    println!("SNLI — Natural Language Inference Example");
    println!("==========================================");
    println!("Vocab: {VOCAB_SIZE}, Embed: {EMBED_DIM}, Hidden: {HIDDEN_SIZE}");
    println!("Seq len: {SEQ_LEN}, Batch: {BATCH_SIZE}, Classes: {NUM_CLASSES}");
    println!();

    let model = SNLIClassifier::new();
    let criterion = CrossEntropyLoss::new();
    let params = model.parameters();
    println!("Model parameters: {}", params.len());

    let mut optimizer = Adam::new(params, LEARNING_RATE);

    let batches_per_epoch = 10;

    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0;
        let mut total_correct = 0usize;
        let mut total_samples = 0usize;

        for _ in 0..batches_per_epoch {
            let (prem_data, hypo_data, label_data) =
                generate_nli_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_CLASSES);

            let premises = Variable::new(Tensor::from_slice(&prem_data, &[BATCH_SIZE, SEQ_LEN]));
            let hypotheses = Variable::new(Tensor::from_slice(&hypo_data, &[BATCH_SIZE, SEQ_LEN]));
            let labels = Variable::new(Tensor::from_slice(&label_data, &[BATCH_SIZE]));

            // Forward pass
            let logits = model.forward(&premises, &hypotheses);

            // Compute loss
            let loss = criterion.forward(&logits, &labels);
            let loss_val = loss.tensor().item().unwrap();

            // Compute accuracy
            let logits_data = logits.tensor().to_vec_f64().unwrap();
            for b in 0..BATCH_SIZE {
                let mut best_class = 0;
                let mut best_val = f64::NEG_INFINITY;
                for c in 0..NUM_CLASSES {
                    let v = logits_data[b * NUM_CLASSES + c];
                    if v > best_val {
                        best_val = v;
                        best_class = c;
                    }
                }
                if best_class == label_data[b] as usize {
                    total_correct += 1;
                }
                total_samples += 1;
            }

            // Backward and update
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss_val;
        }

        let avg_loss = total_loss / batches_per_epoch as f64;
        let accuracy = total_correct as f64 / total_samples as f64 * 100.0;
        println!(
            "Epoch [{:2}/{}] — Loss: {:.4}, Accuracy: {:.1}%",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            accuracy
        );
    }

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("snli_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to snli_model.safetensors");
}
