//! LSTM Word Language Model library.
//!
//! Provides the LSTM language model architecture and helper functions
//! for training and inference. Supports serialization via state_dict.

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, LSTMCell, Linear, Module};

// Hyperparameters (exported for use by binaries)
pub const VOCAB_SIZE: usize = 1000;
pub const EMBED_DIM: usize = 64;
pub const HIDDEN_SIZE: usize = 128;
pub const SEQ_LEN: usize = 35;
pub const BATCH_SIZE: usize = 16;
pub const NUM_EPOCHS: usize = 5;
pub const LEARNING_RATE: f64 = 0.1;

/// LSTM Language Model.
pub struct LSTMLanguageModel {
    pub embedding: Embedding,
    pub lstm_cell: LSTMCell,
    pub decoder: Linear,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl LSTMLanguageModel {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            lstm_cell: LSTMCell::new(embed_dim, hidden_size),
            decoder: Linear::new(hidden_size, vocab_size),
            hidden_size,
            vocab_size,
        }
    }

    /// Forward pass: process a sequence of token indices.
    /// input: [batch_size, seq_len] token indices
    /// Returns: [batch_size * seq_len, vocab_size] logits
    pub fn forward_seq(&self, input: &Variable) -> Variable {
        let batch_size = input.tensor().shape()[0];
        let seq_len = input.tensor().shape()[1];
        let input_data = input.tensor().to_vec_f64().unwrap();

        // Initialize hidden and cell states to zeros
        let mut h = Variable::new(Tensor::zeros(&[batch_size, self.hidden_size]));
        let mut c = Variable::new(Tensor::zeros(&[batch_size, self.hidden_size]));

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
            &[batch_size * seq_len, self.vocab_size],
        ))
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.lstm_cell.parameters());
        params.extend(self.decoder.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // Embedding weight
        let emb_params = self.embedding.parameters();
        sd.insert("embedding.weight".to_string(), emb_params[0].tensor().clone());

        // LSTM parameters: w_ih, w_hh, b_ih, b_hh
        let lstm_params = self.lstm_cell.parameters();
        sd.insert("lstm.w_ih".to_string(), lstm_params[0].tensor().clone());
        sd.insert("lstm.w_hh".to_string(), lstm_params[1].tensor().clone());
        sd.insert("lstm.b_ih".to_string(), lstm_params[2].tensor().clone());
        sd.insert("lstm.b_hh".to_string(), lstm_params[3].tensor().clone());

        // Decoder (fc) parameters
        let fc_params = self.decoder.parameters();
        sd.insert("fc.weight".to_string(), fc_params[0].tensor().clone());
        sd.insert("fc.bias".to_string(), fc_params[1].tensor().clone());

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        let embedding = Embedding::from_tensors(sd["embedding.weight"].clone());

        let lstm_cell = LSTMCell::from_tensors(
            sd["lstm.w_ih"].clone(),
            sd["lstm.w_hh"].clone(),
            sd["lstm.b_ih"].clone(),
            sd["lstm.b_hh"].clone(),
        );

        let decoder = Linear::from_tensors(
            sd["fc.weight"].clone(),
            Some(sd["fc.bias"].clone()),
        );

        let vocab_size = decoder.out_features();
        let hidden_size = decoder.in_features();

        Self {
            embedding,
            lstm_cell,
            decoder,
            hidden_size,
            vocab_size,
        }
    }
}

/// Generate synthetic random token sequences.
/// Returns (input, target) where target is input shifted by one position.
pub fn generate_batch(batch_size: usize, seq_len: usize, vocab_size: usize) -> (Vec<f64>, Vec<f64>) {
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
