//! Language Translation — Transformer-based sequence-to-sequence model.
//!
//! Demonstrates a simplified encoder-decoder Transformer for machine translation.
//! Trains on synthetic data and saves the model to `translation_model.safetensors`.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use language_translation::{
    TranslationModel, generate_parallel_batch,
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBED_DIM, SRC_SEQ_LEN, TGT_SEQ_LEN, BATCH_SIZE,
};

const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.001;

fn main() {
    println!("Language Translation — Transformer Seq2Seq Example");
    println!("====================================================");
    println!("Src vocab: {SRC_VOCAB_SIZE}, Tgt vocab: {TGT_VOCAB_SIZE}, Embed dim: {EMBED_DIM}");
    println!("Src len: {SRC_SEQ_LEN}, Tgt len: {TGT_SEQ_LEN}, Batch: {BATCH_SIZE}");
    println!();

    let model = TranslationModel::new();
    let criterion = CrossEntropyLoss::new();
    let params = model.parameters();
    println!("Model parameters: {}", params.len());

    let mut optimizer = Adam::new(params, LEARNING_RATE);

    let batches_per_epoch = 5;

    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0;

        for _ in 0..batches_per_epoch {
            let (src_data, tgt_in_data, tgt_out_data) = generate_parallel_batch(
                BATCH_SIZE,
                SRC_SEQ_LEN,
                TGT_SEQ_LEN,
                SRC_VOCAB_SIZE,
                TGT_VOCAB_SIZE,
            );

            let src = Variable::new(Tensor::from_slice(&src_data, &[BATCH_SIZE, SRC_SEQ_LEN]));
            let tgt_in = Variable::new(Tensor::from_slice(&tgt_in_data, &[BATCH_SIZE, TGT_SEQ_LEN]));
            let tgt_target = Variable::new(Tensor::from_slice(
                &tgt_out_data,
                &[BATCH_SIZE * TGT_SEQ_LEN],
            ));

            // Forward pass
            let logits = model.forward(&src, &tgt_in);

            // Compute loss
            let loss = criterion.forward(&logits, &tgt_target);
            let loss_val = loss.tensor().item().unwrap();

            // Backward and update
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss_val;
        }

        let avg_loss = total_loss / batches_per_epoch as f64;
        println!(
            "Epoch [{}/{}] — Loss: {:.4}",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss
        );
    }

    // Show a sample translation (input tokens -> output tokens)
    println!();
    println!("Sample translation (greedy decode):");
    let mut rng = rand::thread_rng();
    let sample_src: Vec<f64> = (0..SRC_SEQ_LEN)
        .map(|_| rng.gen_range(1..SRC_VOCAB_SIZE) as f64)
        .collect();
    let sample_tgt_in: Vec<f64> = (0..TGT_SEQ_LEN)
        .map(|_| rng.gen_range(1..TGT_VOCAB_SIZE) as f64)
        .collect();

    let src_var = Variable::new(Tensor::from_slice(&sample_src, &[1, SRC_SEQ_LEN]));
    let tgt_var = Variable::new(Tensor::from_slice(&sample_tgt_in, &[1, TGT_SEQ_LEN]));

    let logits = model.forward(&src_var, &tgt_var);
    let logits_data = logits.tensor().to_vec_f64().unwrap();

    // Greedy decode: argmax at each position
    let mut predicted_tokens = Vec::new();
    for t in 0..TGT_SEQ_LEN {
        let offset = t * TGT_VOCAB_SIZE;
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for v in 0..TGT_VOCAB_SIZE {
            if logits_data[offset + v] > best_val {
                best_val = logits_data[offset + v];
                best_idx = v;
            }
        }
        predicted_tokens.push(best_idx);
    }

    let input_tokens: Vec<usize> = sample_src.iter().map(|&x| x as usize).collect();
    println!("  Input tokens:     {:?}", input_tokens);
    println!("  Predicted tokens: {:?}", predicted_tokens);

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("translation_model.safetensors", bytes).unwrap();

    println!();
    println!("Training complete. Model saved to translation_model.safetensors");
}
