//! Language Translation model definitions.
//!
//! Implements a simplified Transformer-based Seq2Seq model for machine translation.
//! Encoder: Embedding + self-attention (Linear projections) + feedforward Linear
//! Decoder: Embedding + self-attention + cross-attention + output projection Linear

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Embedding, Linear, Module};

// Hyperparameters
pub const SRC_VOCAB_SIZE: usize = 500;
pub const TGT_VOCAB_SIZE: usize = 500;
pub const EMBED_DIM: usize = 64;
pub const NUM_HEADS: usize = 4;
pub const SRC_SEQ_LEN: usize = 10;
pub const TGT_SEQ_LEN: usize = 10;
pub const BATCH_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Attention block with explicit Linear projections (for save/load support)
// ---------------------------------------------------------------------------

pub struct AttentionBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    embed_dim: usize,
}

impl AttentionBlock {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            embed_dim,
        }
    }

    /// Self-attention: query = key = value = input [batch, seq, embed_dim]
    pub fn forward(&self, input: &Variable) -> Variable {
        self.forward_qkv(input, input, input)
    }

    /// Cross-attention: query, key, value can differ.
    pub fn forward_qkv(&self, query: &Variable, key: &Variable, value: &Variable) -> Variable {
        let q = apply_linear_3d(&self.q_proj, query);
        let k = apply_linear_3d(&self.k_proj, key);
        let v = apply_linear_3d(&self.v_proj, value);

        let k_t = k.transpose(-2, -1).unwrap();
        let scale = (self.embed_dim as f64).sqrt();
        let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / scale).unwrap();
        let attn_weights = scores.softmax(-1).unwrap();
        let attn_output = attn_weights.matmul(&v).unwrap();

        apply_linear_3d(&self.out_proj, &attn_output)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.q_proj.named_parameters() {
            sd.insert(format!("{prefix}q_proj.{name}"), param.tensor().clone());
        }
        for (name, param) in self.k_proj.named_parameters() {
            sd.insert(format!("{prefix}k_proj.{name}"), param.tensor().clone());
        }
        for (name, param) in self.v_proj.named_parameters() {
            sd.insert(format!("{prefix}v_proj.{name}"), param.tensor().clone());
        }
        for (name, param) in self.out_proj.named_parameters() {
            sd.insert(format!("{prefix}out_proj.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        let embed_dim = sd[&format!("{prefix}q_proj.weight")].shape()[0];
        Self {
            q_proj: Linear::from_tensors(
                sd[&format!("{prefix}q_proj.weight")].clone(),
                Some(sd[&format!("{prefix}q_proj.bias")].clone()),
            ),
            k_proj: Linear::from_tensors(
                sd[&format!("{prefix}k_proj.weight")].clone(),
                Some(sd[&format!("{prefix}k_proj.bias")].clone()),
            ),
            v_proj: Linear::from_tensors(
                sd[&format!("{prefix}v_proj.weight")].clone(),
                Some(sd[&format!("{prefix}v_proj.bias")].clone()),
            ),
            out_proj: Linear::from_tensors(
                sd[&format!("{prefix}out_proj.weight")].clone(),
                Some(sd[&format!("{prefix}out_proj.bias")].clone()),
            ),
            embed_dim,
        }
    }
}

/// Apply a Linear layer to a 3D input [batch, seq, features].
fn apply_linear_3d(linear: &Linear, input: &Variable) -> Variable {
    let shape = input.tensor().shape();
    let batch = shape[0];
    let seq = shape[1];
    let feat = shape[2];
    let flat = input.reshape(&[batch * seq, feat]).unwrap();
    let out = linear.forward(&flat);
    let out_feat = out.tensor().shape()[1];
    out.reshape(&[batch, seq, out_feat]).unwrap()
}

// ---------------------------------------------------------------------------
// Transformer Encoder
// ---------------------------------------------------------------------------

pub struct TransformerEncoder {
    pub embedding: Embedding,
    pub self_attn: AttentionBlock,
    pub fc: Linear,
}

impl TransformerEncoder {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            self_attn: AttentionBlock::new(embed_dim),
            fc: Linear::new(embed_dim, embed_dim),
        }
    }

    /// Forward: src_tokens [batch, src_len] -> encoder output [batch, src_len, embed_dim]
    pub fn forward_encode(&self, src: &Variable) -> Variable {
        let embedded = self.embedding.forward(src);
        let attn_out = self.self_attn.forward(&embedded);
        let residual = attn_out.add(&embedded).unwrap();

        let shape = residual.tensor().shape().to_vec();
        let batch = shape[0];
        let seq = shape[1];
        let feat = shape[2];
        let flat = residual.reshape(&[batch * seq, feat]).unwrap();
        let fc_out = self.fc.forward(&flat);
        fc_out.reshape(&[batch, seq, feat]).unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.fc.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        let emb_params = self.embedding.parameters();
        sd.insert(
            format!("{prefix}embedding.weight"),
            emb_params[0].tensor().clone(),
        );
        sd.extend(self.self_attn.state_dict(&format!("{prefix}attention.")));
        for (name, param) in self.fc.named_parameters() {
            sd.insert(format!("{prefix}fc.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        let _embed_dim = sd[&format!("{prefix}fc.weight")].shape()[0];
        Self {
            embedding: Embedding::from_tensors(
                sd[&format!("{prefix}embedding.weight")].clone(),
            ),
            self_attn: AttentionBlock::from_state_dict(sd, &format!("{prefix}attention.")),
            fc: Linear::from_tensors(
                sd[&format!("{prefix}fc.weight")].clone(),
                Some(sd[&format!("{prefix}fc.bias")].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer Decoder
// ---------------------------------------------------------------------------

pub struct TransformerDecoder {
    pub embedding: Embedding,
    pub self_attn: AttentionBlock,
    pub cross_attn: AttentionBlock,
    pub output_proj: Linear,
}

impl TransformerDecoder {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, embed_dim),
            self_attn: AttentionBlock::new(embed_dim),
            cross_attn: AttentionBlock::new(embed_dim),
            output_proj: Linear::new(embed_dim, vocab_size),
        }
    }

    /// Forward: tgt_tokens [batch, tgt_len], encoder_out [batch, src_len, embed_dim]
    /// Returns: logits [batch * tgt_len, vocab_size]
    pub fn forward_decode(&self, tgt: &Variable, encoder_out: &Variable) -> Variable {
        let shape = tgt.tensor().shape().to_vec();
        let batch = shape[0];
        let tgt_len = shape[1];

        let embedded = self.embedding.forward(tgt);
        let self_attn_out = self.self_attn.forward(&embedded);
        let residual1 = self_attn_out.add(&embedded).unwrap();

        let cross_attn_out = self.cross_attn.forward_qkv(&residual1, encoder_out, encoder_out);
        let residual2 = cross_attn_out.add(&residual1).unwrap();

        let embed_dim = residual2.tensor().shape()[2];
        let flat = residual2.reshape(&[batch * tgt_len, embed_dim]).unwrap();
        self.output_proj.forward(&flat)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.embedding.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.output_proj.parameters());
        params
    }

    pub fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        let emb_params = self.embedding.parameters();
        sd.insert(
            format!("{prefix}embedding.weight"),
            emb_params[0].tensor().clone(),
        );
        sd.extend(self.self_attn.state_dict(&format!("{prefix}self_attn.")));
        sd.extend(self.cross_attn.state_dict(&format!("{prefix}cross_attn.")));
        for (name, param) in self.output_proj.named_parameters() {
            sd.insert(format!("{prefix}output_proj.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str) -> Self {
        Self {
            embedding: Embedding::from_tensors(
                sd[&format!("{prefix}embedding.weight")].clone(),
            ),
            self_attn: AttentionBlock::from_state_dict(sd, &format!("{prefix}self_attn.")),
            cross_attn: AttentionBlock::from_state_dict(sd, &format!("{prefix}cross_attn.")),
            output_proj: Linear::from_tensors(
                sd[&format!("{prefix}output_proj.weight")].clone(),
                Some(sd[&format!("{prefix}output_proj.bias")].clone()),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Full Seq2Seq Transformer model
// ---------------------------------------------------------------------------

pub struct TranslationModel {
    pub encoder: TransformerEncoder,
    pub decoder: TransformerDecoder,
}

impl TranslationModel {
    pub fn new() -> Self {
        Self {
            encoder: TransformerEncoder::new(SRC_VOCAB_SIZE, EMBED_DIM),
            decoder: TransformerDecoder::new(TGT_VOCAB_SIZE, EMBED_DIM),
        }
    }

    pub fn forward(&self, src: &Variable, tgt: &Variable) -> Variable {
        let enc_out = self.encoder.forward_encode(src);
        self.decoder.forward_decode(tgt, &enc_out)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = self.encoder.state_dict("encoder.");
        sd.extend(self.decoder.state_dict("decoder."));
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            encoder: TransformerEncoder::from_state_dict(sd, "encoder."),
            decoder: TransformerDecoder::from_state_dict(sd, "decoder."),
        }
    }
}

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/// Generate synthetic parallel sentence pairs.
/// Returns (src_tokens, tgt_input, tgt_target).
pub fn generate_parallel_batch(
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
        for _ in 0..src_len {
            src.push(rng.gen_range(1..src_vocab) as f64);
        }
        let tgt_tokens: Vec<f64> = (0..tgt_len + 1)
            .map(|_| rng.gen_range(1..tgt_vocab) as f64)
            .collect();
        tgt_in.extend_from_slice(&tgt_tokens[..tgt_len]);
        tgt_out.extend_from_slice(&tgt_tokens[1..]);
    }

    (src, tgt_in, tgt_out)
}
