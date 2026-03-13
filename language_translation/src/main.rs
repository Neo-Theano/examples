//! Language Translation — Transformer-based sequence-to-sequence model.
//!
//! Demonstrates a simplified encoder-decoder Transformer for machine translation.
//! Encoder: Embedding + MultiheadAttention + Linear
//! Decoder: Embedding + MultiheadAttention (self + cross) + Linear
//! Uses synthetic parallel sentence pairs (random token sequences).

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, Linear, Module, MultiheadAttention, CrossEntropyLoss};
use theano_optim::{Adam, Optimizer};

// Hyperparameters
const SRC_VOCAB_SIZE: usize = 500;
const TGT_VOCAB_SIZE: usize = 500;
const EMBED_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const SRC_SEQ_LEN: usize = 10;
const TGT_SEQ_LEN: usize = 10;
const BATCH_SIZE: usize = 8;
const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 0.001;

/// Transformer Encoder block.
struct TransformerEncoder {
    embedding: Embedding,
    self_attn: MultiheadAttention,
    fc: Linear,
}

impl TransformerEncoder {
    fn new(vocab_size: usize, embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            self_attn: MultiheadAttention::new(embed_dim, num_heads),
            fc: Linear::new(embed_dim, embed_dim),
        }
    }

    /// Forward: src_tokens [batch, src_len] -> encoder output [batch, src_len, embed_dim]
    fn forward_encode(&self, src: &Variable) -> Variable {
        // Embed source tokens: [batch, src_len] -> [batch, src_len, embed_dim]
        let embedded = self.embedding.forward(src);

        // Self-attention: [batch, src_len, embed_dim]
        let attn_out = self.self_attn.forward(&embedded);

        // Residual connection + feedforward
        let residual = attn_out.add(&embedded).unwrap();

        // Apply feedforward to each position (reshape to 2D, apply, reshape back)
        let shape = residual.tensor().shape().to_vec();
        let batch = shape[0];
        let seq = shape[1];
        let feat = shape[2];
        let flat = residual.reshape(&[batch * seq, feat]).unwrap();
        let fc_out = self.fc.forward(&flat);
        fc_out.reshape(&[batch, seq, feat]).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.fc.parameters());
        params
    }
}

/// Transformer Decoder block.
struct TransformerDecoder {
    embedding: Embedding,
    self_attn: MultiheadAttention,
    cross_attn: MultiheadAttention,
    output_proj: Linear,
}

impl TransformerDecoder {
    fn new(vocab_size: usize, embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            self_attn: MultiheadAttention::new(embed_dim, num_heads),
            cross_attn: MultiheadAttention::new(embed_dim, num_heads),
            output_proj: Linear::new(embed_dim, vocab_size),
        }
    }

    /// Forward: tgt_tokens [batch, tgt_len], encoder_out [batch, src_len, embed_dim]
    /// Returns: logits [batch * tgt_len, vocab_size]
    fn forward_decode(&self, tgt: &Variable, encoder_out: &Variable) -> Variable {
        let shape = tgt.tensor().shape().to_vec();
        let batch = shape[0];
        let tgt_len = shape[1];

        // Embed target tokens: [batch, tgt_len] -> [batch, tgt_len, embed_dim]
        let embedded = self.embedding.forward(tgt);

        // Self-attention on target
        let self_attn_out = self.self_attn.forward(&embedded);
        let residual1 = self_attn_out.add(&embedded).unwrap();

        // Cross-attention: query=decoder, key/value=encoder
        let cross_attn_out = self.cross_attn.forward_qkv(&residual1, encoder_out, encoder_out);
        let residual2 = cross_attn_out.add(&residual1).unwrap();

        // Project to vocabulary: flatten to [batch * tgt_len, embed_dim]
        let flat = residual2.reshape(&[batch * tgt_len, EMBED_DIM]).unwrap();
        self.output_proj.forward(&flat)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.output_proj.parameters());
        params
    }
}

/// Full Seq2Seq Transformer model.
struct Seq2SeqTransformer {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
}

impl Seq2SeqTransformer {
    fn new() -> Self {
        Self {
            encoder: TransformerEncoder::new(SRC_VOCAB_SIZE, EMBED_DIM, NUM_HEADS),
            decoder: TransformerDecoder::new(TGT_VOCAB_SIZE, EMBED_DIM, NUM_HEADS),
        }
    }

    fn forward(&self, src: &Variable, tgt: &Variable) -> Variable {
        let enc_out = self.encoder.forward_encode(src);
        self.decoder.forward_decode(tgt, &enc_out)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}

/// Generate synthetic parallel sentence pairs.
/// Returns (src_tokens, tgt_input, tgt_target).
/// tgt_target is tgt_input shifted by one (simplified).
fn generate_parallel_batch(
    batch_size: usize,
    src_len: usize,
    tgt_len: usize,
    src_vocab: usize,
    tgt_vocab: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut src = Vec::with_capacity(batch_size * src_len);
    let mut tgt_in = Vec::with_capacity(batch_size * tgt_len);
    let mut tgt_out = Vec::with_capacity(batch_size * tgt_len);

    for _ in 0..batch_size {
        // Source sentence: random tokens
        for _ in 0..src_len {
            src.push(rng.gen_range(1..src_vocab) as f64);
        }
        // Target sentence: random tokens (shifted for teacher forcing)
        let tgt_tokens: Vec<f64> = (0..tgt_len + 1)
            .map(|_| rng.gen_range(1..tgt_vocab) as f64)
            .collect();
        tgt_in.extend_from_slice(&tgt_tokens[..tgt_len]);
        tgt_out.extend_from_slice(&tgt_tokens[1..]);
    }

    (src, tgt_in, tgt_out)
}

fn main() {
    println!("Language Translation — Transformer Seq2Seq Example");
    println!("====================================================");
    println!("Src vocab: {SRC_VOCAB_SIZE}, Tgt vocab: {TGT_VOCAB_SIZE}, Embed dim: {EMBED_DIM}");
    println!("Src len: {SRC_SEQ_LEN}, Tgt len: {TGT_SEQ_LEN}, Batch: {BATCH_SIZE}");
    println!();

    let model = Seq2SeqTransformer::new();
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
    println!();
    println!("Training complete.");
}
