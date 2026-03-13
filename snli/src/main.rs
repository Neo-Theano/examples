//! SNLI — Natural Language Inference example.
//!
//! Sentence pair classification: premise + hypothesis -> entailment/contradiction/neutral.
//! Model: Embedding -> BiLSTM encoder (forward + reverse LSTMCell) ->
//!        concatenate representations -> Linear(4*hidden, 3)
//! Uses synthetic data with 3-class labels. Prints accuracy per epoch.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, LSTMCell, Linear, Module, CrossEntropyLoss};
use theano_optim::{Adam, Optimizer};

// Hyperparameters
const VOCAB_SIZE: usize = 500;
const EMBED_DIM: usize = 64;
const HIDDEN_SIZE: usize = 64;
const NUM_CLASSES: usize = 3; // entailment, contradiction, neutral
const SEQ_LEN: usize = 12;
const BATCH_SIZE: usize = 16;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 0.001;

/// BiLSTM Encoder: runs forward and reverse LSTMCell over a sequence,
/// returns the concatenation of final hidden states: [batch, 2*hidden].
struct BiLSTMEncoder {
    embedding: Embedding,
    forward_cell: LSTMCell,
    reverse_cell: LSTMCell,
}

impl BiLSTMEncoder {
    fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            forward_cell: LSTMCell::new(embed_dim, hidden_size),
            reverse_cell: LSTMCell::new(embed_dim, hidden_size),
        }
    }

    /// Encode a batch of token sequences.
    /// tokens: [batch, seq_len] -> output: [batch, 2*hidden_size]
    fn encode(&self, tokens: &Variable) -> Variable {
        let shape = tokens.tensor().shape().to_vec();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let tokens_data = tokens.tensor().to_vec_f64().unwrap();

        // Embed all tokens at once: [batch, seq_len] -> [batch, seq_len, embed_dim]
        // We process step by step for the LSTM cells

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

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.forward_cell.parameters());
        params.extend(self.reverse_cell.parameters());
        params
    }
}

/// SNLI Model: encodes premise and hypothesis with BiLSTM,
/// concatenates their representations, and classifies with a linear layer.
struct SNLIModel {
    encoder: BiLSTMEncoder,
    classifier: Linear,
}

impl SNLIModel {
    fn new() -> Self {
        Self {
            encoder: BiLSTMEncoder::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE),
            // Input: concatenation of premise_repr and hypothesis_repr
            // Each is [batch, 2*hidden], so concat is [batch, 4*hidden]
            classifier: Linear::new(4 * HIDDEN_SIZE, NUM_CLASSES),
        }
    }

    /// Forward pass.
    /// premise: [batch, seq_len], hypothesis: [batch, seq_len]
    /// Returns: logits [batch, NUM_CLASSES]
    fn forward(&self, premise: &Variable, hypothesis: &Variable) -> Variable {
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

        // Classify
        self.classifier.forward(&concat)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.classifier.parameters());
        params
    }
}

/// Generate synthetic NLI data.
/// Returns (premises, hypotheses, labels).
fn generate_nli_batch(
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

fn main() {
    println!("SNLI — Natural Language Inference Example");
    println!("==========================================");
    println!("Vocab: {VOCAB_SIZE}, Embed: {EMBED_DIM}, Hidden: {HIDDEN_SIZE}");
    println!("Seq len: {SEQ_LEN}, Batch: {BATCH_SIZE}, Classes: {NUM_CLASSES}");
    println!();

    let model = SNLIModel::new();
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

    println!();
    println!("Training complete.");
}
