//! SNLI model definitions.
//!
//! Implements a Natural Language Inference model:
//!   Embedding -> BiLSTM encoder (forward + reverse LSTMCell) ->
//!   concatenate representations -> Linear(4*hidden, 3)

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, LSTMCell, Linear, Module};

// Hyperparameters
pub const VOCAB_SIZE: usize = 500;
pub const EMBED_DIM: usize = 64;
pub const HIDDEN_SIZE: usize = 64;
pub const NUM_CLASSES: usize = 3; // entailment, contradiction, neutral
pub const SEQ_LEN: usize = 12;
pub const BATCH_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// BiLSTM Encoder
// ---------------------------------------------------------------------------

pub struct BiLSTMEncoder {
    pub embedding: Embedding,
    pub forward_cell: LSTMCell,
    pub reverse_cell: LSTMCell,
}

impl BiLSTMEncoder {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            forward_cell: LSTMCell::new(embed_dim, hidden_size),
            reverse_cell: LSTMCell::new(embed_dim, hidden_size),
        }
    }

    /// Encode a batch of token sequences.
    /// tokens: [batch, seq_len] -> output: [batch, 2*hidden_size]
    pub fn encode(&self, tokens: &Variable) -> Variable {
        let shape = tokens.tensor().shape().to_vec();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let tokens_data = tokens.tensor().to_vec_f64().unwrap();

        // Forward LSTM
        let mut fwd_h = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut fwd_c = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));

        for t in 0..seq_len {
            let mut step_tokens = vec![0.0f64; batch_size];
            for b in 0..batch_size {
                step_tokens[b] = tokens_data[b * seq_len + t];
            }
            let step_input = Variable::new(Tensor::from_slice(&step_tokens, &[batch_size]));
            let embedded = self.embedding.forward(&step_input);
            let (new_h, new_c) = self.forward_cell.forward_cell(&embedded, &fwd_h, &fwd_c);
            fwd_h = new_h;
            fwd_c = new_c;
        }

        // Reverse LSTM (iterate from end to start)
        let mut rev_h = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));
        let mut rev_c = Variable::new(Tensor::zeros(&[batch_size, HIDDEN_SIZE]));

        for t in (0..seq_len).rev() {
            let mut step_tokens = vec![0.0f64; batch_size];
            for b in 0..batch_size {
                step_tokens[b] = tokens_data[b * seq_len + t];
            }
            let step_input = Variable::new(Tensor::from_slice(&step_tokens, &[batch_size]));
            let embedded = self.embedding.forward(&step_input);
            let (new_h, new_c) = self.reverse_cell.forward_cell(&embedded, &rev_h, &rev_c);
            rev_h = new_h;
            rev_c = new_c;
        }

        // Concatenate forward and reverse final hidden states: [batch, 2*hidden]
        let fwd_data = fwd_h.tensor().to_vec_f64().unwrap();
        let rev_data = rev_h.tensor().to_vec_f64().unwrap();
        let mut concat_data = Vec::with_capacity(batch_size * 2 * HIDDEN_SIZE);
        for b in 0..batch_size {
            concat_data.extend_from_slice(&fwd_data[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE]);
            concat_data.extend_from_slice(&rev_data[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE]);
        }

        Variable::new(Tensor::from_slice(&concat_data, &[batch_size, 2 * HIDDEN_SIZE]))
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.forward_cell.parameters());
        params.extend(self.reverse_cell.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // Embedding weight
        let emb_params = self.embedding.parameters();
        sd.insert(
            format!("{prefix}embedding.weight"),
            emb_params[0].tensor().clone(),
        );

        // Forward LSTM parameters: [w_ih, w_hh, b_ih, b_hh]
        let fwd_params = self.forward_cell.parameters();
        sd.insert(format!("{prefix}fwd_lstm.w_ih"), fwd_params[0].tensor().clone());
        sd.insert(format!("{prefix}fwd_lstm.w_hh"), fwd_params[1].tensor().clone());
        sd.insert(format!("{prefix}fwd_lstm.b_ih"), fwd_params[2].tensor().clone());
        sd.insert(format!("{prefix}fwd_lstm.b_hh"), fwd_params[3].tensor().clone());

        // Reverse LSTM parameters
        let rev_params = self.reverse_cell.parameters();
        sd.insert(format!("{prefix}rev_lstm.w_ih"), rev_params[0].tensor().clone());
        sd.insert(format!("{prefix}rev_lstm.w_hh"), rev_params[1].tensor().clone());
        sd.insert(format!("{prefix}rev_lstm.b_ih"), rev_params[2].tensor().clone());
        sd.insert(format!("{prefix}rev_lstm.b_hh"), rev_params[3].tensor().clone());

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        Self {
            embedding: Embedding::from_tensors(
                sd[&format!("{prefix}embedding.weight")].clone(),
            ),
            forward_cell: LSTMCell::from_tensors(
                sd[&format!("{prefix}fwd_lstm.w_ih")].clone(),
                sd[&format!("{prefix}fwd_lstm.w_hh")].clone(),
                sd[&format!("{prefix}fwd_lstm.b_ih")].clone(),
                sd[&format!("{prefix}fwd_lstm.b_hh")].clone(),
            ),
            reverse_cell: LSTMCell::from_tensors(
                sd[&format!("{prefix}rev_lstm.w_ih")].clone(),
                sd[&format!("{prefix}rev_lstm.w_hh")].clone(),
                sd[&format!("{prefix}rev_lstm.b_ih")].clone(),
                sd[&format!("{prefix}rev_lstm.b_hh")].clone(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// SNLI Classifier
// ---------------------------------------------------------------------------

pub struct SNLIClassifier {
    pub encoder: BiLSTMEncoder,
    pub classifier: Linear,
}

impl SNLIClassifier {
    pub fn new() -> Self {
        Self {
            encoder: BiLSTMEncoder::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE),
            classifier: Linear::new(4 * HIDDEN_SIZE, NUM_CLASSES),
        }
    }

    /// Forward pass.
    /// premise: [batch, seq_len], hypothesis: [batch, seq_len]
    /// Returns: logits [batch, NUM_CLASSES]
    pub fn forward(&self, premise: &Variable, hypothesis: &Variable) -> Variable {
        let batch_size = premise.tensor().shape()[0];

        let prem_repr = self.encoder.encode(premise);    // [batch, 2*hidden]
        let hypo_repr = self.encoder.encode(hypothesis); // [batch, 2*hidden]

        // Concatenate premise and hypothesis representations: [batch, 4*hidden]
        let prem_data = prem_repr.tensor().to_vec_f64().unwrap();
        let hypo_data = hypo_repr.tensor().to_vec_f64().unwrap();
        let repr_dim = 2 * HIDDEN_SIZE;

        let mut concat_data = Vec::with_capacity(batch_size * 4 * HIDDEN_SIZE);
        for b in 0..batch_size {
            concat_data.extend_from_slice(&prem_data[b * repr_dim..(b + 1) * repr_dim]);
            concat_data.extend_from_slice(&hypo_data[b * repr_dim..(b + 1) * repr_dim]);
        }
        let concat = Variable::new(Tensor::from_slice(
            &concat_data,
            &[batch_size, 4 * HIDDEN_SIZE],
        ));

        self.classifier.forward(&concat)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.classifier.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = self.encoder.state_dict("encoder.");
        for (name, param) in self.classifier.named_parameters() {
            sd.insert(format!("classifier.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            encoder: BiLSTMEncoder::from_state_dict(sd, "encoder."),
            classifier: Linear::from_tensors(
                sd["classifier.weight"].clone(),
                Some(sd["classifier.bias"].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/// Generate synthetic NLI data.
/// Returns (premises, hypotheses, labels).
pub fn generate_nli_batch(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    num_classes: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut premises = Vec::with_capacity(batch_size * seq_len);
    let mut hypotheses = Vec::with_capacity(batch_size * seq_len);
    let mut labels = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        for _ in 0..seq_len {
            premises.push(rng.gen_range(1..vocab_size) as f64);
            hypotheses.push(rng.gen_range(1..vocab_size) as f64);
        }
        labels.push(rng.gen_range(0..num_classes) as f64);
    }

    (premises, hypotheses, labels)
}
