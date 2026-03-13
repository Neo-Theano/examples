//! Word Language Model — RNN/LSTM language model example.
//!
//! Demonstrates training an LSTM-based language model on synthetic token sequences.
//! Model: Embedding(vocab_size, embed_dim) -> LSTM cells (iterate over sequence) -> Linear(hidden, vocab_size)
//! Prints perplexity (exp(loss)) per epoch.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, LSTMCell, Linear, Module, CrossEntropyLoss};
use theano_optim::{SGD, Optimizer};

// Hyperparameters
const VOCAB_SIZE: usize = 1000;
const EMBED_DIM: usize = 64;
const HIDDEN_SIZE: usize = 128;
const SEQ_LEN: usize = 35;
const BATCH_SIZE: usize = 16;
const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.1;

/// LSTM Language Model.
struct LSTMLanguageModel {
    embedding: Embedding,
    lstm_cell: LSTMCell,
    decoder: Linear,
}

impl LSTMLanguageModel {
    fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            lstm_cell: LSTMCell::new(embed_dim, hidden_size),
            decoder: Linear::new(hidden_size, vocab_size),
        }
    }

    /// Forward pass: process a sequence of token indices.
    /// input: [batch_size, seq_len] token indices
    /// Returns: [batch_size * seq_len, vocab_size] logits
    fn forward_seq(&self, input: &Variable) -> Variable {
        let batch_size = input.tensor().shape()[0];
        let seq_len = input.tensor().shape()[1];
        let input_data = input.tensor().to_vec_f64().unwrap();

        // Initialize hidden and cell states to zeros
        let mut h = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut c = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));

        // Collect outputs from each timestep
        let mut all_outputs: Vec<f64> = Vec::new();

        for t in 0..seq_len {
            // Extract token indices at timestep t: [batch_size]
            let mut step_tokens = vec![0.0f64; batch_size];
            for b in 0..batch_size {
                step_tokens[b] = input_data[b * seq_len + t];
            }
            let step_input = Variable::new(Tensor::from_slice(&step_tokens, &[batch_size]));

            // Embed: [batch_size] -> [batch_size, embed_dim]
            let embedded = self.embedding.forward(&step_input);

            // LSTM cell step: (embedded, h, c) -> (new_h, new_c)
            let (new_h, new_c) = self.lstm_cell.forward_cell(&embedded, &h, &c);
            h = new_h;
            c = new_c;

            // Decode hidden state: [batch_size, hidden_size] -> [batch_size, vocab_size]
            let output = self.decoder.forward(&h);
            let out_data = output.tensor().to_vec_f64().unwrap();
            all_outputs.extend_from_slice(&out_data);
        }

        // Stack all outputs: [batch_size * seq_len, vocab_size]
        Variable::new(Tensor::from_slice(
            &all_outputs,
            &[batch_size * seq_len, VOCAB_SIZE],
        ))
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.lstm_cell.parameters());
        params.extend(self.decoder.parameters());
        params
    }
}

/// Generate synthetic random token sequences.
/// Returns (input, target) where target is input shifted by one position.
fn generate_batch(batch_size: usize, seq_len: usize, vocab_size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let total_len = seq_len + 1; // +1 for target shift
    let mut input = Vec::with_capacity(batch_size * seq_len);
    let mut target = Vec::with_capacity(batch_size * seq_len);

    for _ in 0..batch_size {
        let tokens: Vec<f64> = (0..total_len)
            .map(|_| rng.gen_range(0..vocab_size) as f64)
            .collect();
        // Input is tokens[0..seq_len], target is tokens[1..seq_len+1]
        input.extend_from_slice(&tokens[..seq_len]);
        target.extend_from_slice(&tokens[1..]);
    }

    (input, target)
}

fn main() {
    println!("Word Language Model — LSTM Language Model Example");
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
            "Epoch [{}/{}] — Loss: {:.4}, Perplexity: {:.2}",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            perplexity
        );
    }

    println!();
    println!("Training complete.");
}
