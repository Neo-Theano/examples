//! Word Language Model -- LSTM Language Model Example.
//!
//! Demonstrates training an LSTM-based language model on synthetic token sequences.
//! Model: Embedding(vocab_size, embed_dim) -> LSTM cells (iterate over sequence) -> Linear(hidden, vocab_size)
//! Prints perplexity (exp(loss)) per epoch.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::CrossEntropyLoss;
use theano_optim::{SGD, Optimizer};
use theano_serialize::save_state_dict;

use word_language_model::{
    LSTMLanguageModel, generate_batch,
    VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, SEQ_LEN, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
};

fn main() {
    println!("Word Language Model -- LSTM Language Model Example");
    println!("==================================================");
    println!("Vocab size: {VOCAB_SIZE}, Embed dim: {EMBED_DIM}, Hidden: {HIDDEN_SIZE}");
    println!("Seq length: {SEQ_LEN}, Batch size: {BATCH_SIZE}");
    println!();

    let model = LSTMLanguageModel::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE);
    let criterion = CrossEntropyLoss::new();
    let params = model.parameters();
    println!("Model parameters: {}", params.len());

    let mut optimizer = SGD::new(params, LEARNING_RATE);

    let batches_per_epoch = 5;

    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for _ in 0..batches_per_epoch {
            // Generate synthetic data
            let (input_data, target_data) = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);

            let input = Variable::new(Tensor::from_slice(&input_data, &[BATCH_SIZE, SEQ_LEN]));
            let target = Variable::new(Tensor::from_slice(
                &target_data,
                &[BATCH_SIZE * SEQ_LEN],
            ));

            // Forward pass: get logits [batch*seq_len, vocab_size]
            let logits = model.forward_seq(&input);

            // Compute cross-entropy loss
            let loss = criterion.forward(&logits, &target);
            let loss_val = loss.tensor().item().unwrap();

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss_val;
            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f64;
        let perplexity = avg_loss.exp();
        println!(
            "Epoch [{}/{}] -- Loss: {:.4}, Perplexity: {:.2}",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            perplexity
        );
    }

    // Save model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("word_lm_model.safetensors", &bytes).expect("failed to save model");
    println!("\nModel saved to word_lm_model.safetensors ({} bytes)", bytes.len());

    println!();
    println!("Training complete.");
}
