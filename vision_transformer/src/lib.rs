//! Vision Transformer (ViT) model library.
//!
//! Provides the ViT architecture and helper functions for training
//! and inference. Supports serialization via state_dict.

use std::collections::HashMap;

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{
    LayerNorm, Linear, MultiheadAttention, Module,
};

// ---------------------------------------------------------------------------
// Patch Embedding
//
// Splits an image [N, C, H, W] into non-overlapping patches of size P x P
// and projects each patch to embed_dim via a linear layer.
// Output: [N, num_patches, embed_dim]
// ---------------------------------------------------------------------------
pub struct PatchEmbedding {
    pub patch_size: usize,
    pub projection: Linear,
    pub num_patches: usize,
}

impl PatchEmbedding {
    pub fn new(
        img_channels: usize,
        img_size: usize,
        patch_size: usize,
        embed_dim: usize,
    ) -> Self {
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let patch_dim = img_channels * patch_size * patch_size;
        Self {
            patch_size,
            projection: Linear::new(patch_dim, embed_dim),
            num_patches,
        }
    }

    pub fn from_state_dict(
        sd: &HashMap<String, Tensor>,
        _img_channels: usize,
        img_size: usize,
        patch_size: usize,
    ) -> Self {
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        let projection = Linear::from_tensors(
            sd["patch_embed.weight"].clone(),
            Some(sd["patch_embed.bias"].clone()),
        );
        Self {
            patch_size,
            projection,
            num_patches,
        }
    }

    pub fn state_dict_entries(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        let params = self.projection.parameters();
        sd.insert("patch_embed.weight".to_string(), params[0].tensor().clone());
        sd.insert("patch_embed.bias".to_string(), params[1].tensor().clone());
        sd
    }
}

impl Module for PatchEmbedding {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape().to_vec();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let p = self.patch_size;
        let ph = h / p;
        let pw = w / p;
        let patch_dim = c * p * p;

        let data = input.tensor().to_vec_f64().unwrap();
        let num_patches = ph * pw;
        let mut patches = vec![0.0f64; n * num_patches * patch_dim];

        for batch in 0..n {
            for py in 0..ph {
                for px in 0..pw {
                    let patch_idx = py * pw + px;
                    let mut offset = 0;
                    for ch in 0..c {
                        for dy in 0..p {
                            for dx in 0..p {
                                let iy = py * p + dy;
                                let ix = px * p + dx;
                                let src = batch * c * h * w + ch * h * w + iy * w + ix;
                                let dst =
                                    batch * num_patches * patch_dim + patch_idx * patch_dim + offset;
                                patches[dst] = data[src];
                                offset += 1;
                            }
                        }
                    }
                }
            }
        }

        // [N, num_patches, patch_dim]
        let patches_var =
            Variable::new(Tensor::from_slice(&patches, &[n, num_patches, patch_dim]));

        // Project each patch: reshape to [N * num_patches, patch_dim], apply linear, reshape back
        let flat = patches_var
            .reshape(&[n * num_patches, patch_dim])
            .unwrap();
        let projected = self.projection.forward(&flat);
        let embed_dim = projected.tensor().shape()[1];
        projected.reshape(&[n, num_patches, embed_dim]).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        self.projection.parameters()
    }
}

// ---------------------------------------------------------------------------
// Transformer Block
//   LayerNorm -> MultiheadAttention -> residual -> LayerNorm -> MLP -> residual
// ---------------------------------------------------------------------------
pub struct TransformerBlock {
    pub norm1: LayerNorm,
    pub attn: MultiheadAttention,
    pub norm2: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_dim: usize) -> Self {
        Self {
            norm1: LayerNorm::new(vec![embed_dim]),
            attn: MultiheadAttention::new(embed_dim, num_heads),
            norm2: LayerNorm::new(vec![embed_dim]),
            mlp_fc1: Linear::new(embed_dim, mlp_dim),
            mlp_fc2: Linear::new(mlp_dim, embed_dim),
        }
    }

    pub fn state_dict_entries(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // norm1
        let norm1_params = self.norm1.parameters();
        sd.insert(format!("{prefix}norm1.weight"), norm1_params[0].tensor().clone());
        sd.insert(format!("{prefix}norm1.bias"), norm1_params[1].tensor().clone());

        // attn (q_proj, k_proj, v_proj, out_proj)
        let attn_params = self.attn.parameters();
        sd.insert(format!("{prefix}attn.q_proj.weight"), attn_params[0].tensor().clone());
        sd.insert(format!("{prefix}attn.q_proj.bias"), attn_params[1].tensor().clone());
        sd.insert(format!("{prefix}attn.k_proj.weight"), attn_params[2].tensor().clone());
        sd.insert(format!("{prefix}attn.k_proj.bias"), attn_params[3].tensor().clone());
        sd.insert(format!("{prefix}attn.v_proj.weight"), attn_params[4].tensor().clone());
        sd.insert(format!("{prefix}attn.v_proj.bias"), attn_params[5].tensor().clone());
        sd.insert(format!("{prefix}attn.out_proj.weight"), attn_params[6].tensor().clone());
        sd.insert(format!("{prefix}attn.out_proj.bias"), attn_params[7].tensor().clone());

        // norm2
        let norm2_params = self.norm2.parameters();
        sd.insert(format!("{prefix}norm2.weight"), norm2_params[0].tensor().clone());
        sd.insert(format!("{prefix}norm2.bias"), norm2_params[1].tensor().clone());

        // mlp
        let fc1_params = self.mlp_fc1.parameters();
        sd.insert(format!("{prefix}mlp.fc1.weight"), fc1_params[0].tensor().clone());
        sd.insert(format!("{prefix}mlp.fc1.bias"), fc1_params[1].tensor().clone());

        let fc2_params = self.mlp_fc2.parameters();
        sd.insert(format!("{prefix}mlp.fc2.weight"), fc2_params[0].tensor().clone());
        sd.insert(format!("{prefix}mlp.fc2.bias"), fc2_params[1].tensor().clone());

        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>, prefix: &str, num_heads: usize) -> Self {
        let norm1 = LayerNorm::from_tensors(
            sd[&format!("{prefix}norm1.weight")].clone(),
            sd[&format!("{prefix}norm1.bias")].clone(),
        );

        let q_proj = Linear::from_tensors(
            sd[&format!("{prefix}attn.q_proj.weight")].clone(),
            Some(sd[&format!("{prefix}attn.q_proj.bias")].clone()),
        );
        let k_proj = Linear::from_tensors(
            sd[&format!("{prefix}attn.k_proj.weight")].clone(),
            Some(sd[&format!("{prefix}attn.k_proj.bias")].clone()),
        );
        let v_proj = Linear::from_tensors(
            sd[&format!("{prefix}attn.v_proj.weight")].clone(),
            Some(sd[&format!("{prefix}attn.v_proj.bias")].clone()),
        );
        let out_proj = Linear::from_tensors(
            sd[&format!("{prefix}attn.out_proj.weight")].clone(),
            Some(sd[&format!("{prefix}attn.out_proj.bias")].clone()),
        );
        let attn = MultiheadAttention::from_linears(num_heads, q_proj, k_proj, v_proj, out_proj);

        let norm2 = LayerNorm::from_tensors(
            sd[&format!("{prefix}norm2.weight")].clone(),
            sd[&format!("{prefix}norm2.bias")].clone(),
        );

        let mlp_fc1 = Linear::from_tensors(
            sd[&format!("{prefix}mlp.fc1.weight")].clone(),
            Some(sd[&format!("{prefix}mlp.fc1.bias")].clone()),
        );
        let mlp_fc2 = Linear::from_tensors(
            sd[&format!("{prefix}mlp.fc2.weight")].clone(),
            Some(sd[&format!("{prefix}mlp.fc2.bias")].clone()),
        );

        Self { norm1, attn, norm2, mlp_fc1, mlp_fc2 }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Variable) -> Variable {
        // Pre-norm self-attention with residual
        let normed = self.norm1.forward(input);
        let attn_out = self.attn.forward(&normed);
        let x = input.add(&attn_out).unwrap();

        // Pre-norm MLP with residual
        let normed2 = self.norm2.forward(&x);

        // MLP: flatten to 2D, apply layers, reshape back
        let shape = normed2.tensor().shape().to_vec();
        let (batch, seq, feat) = (shape[0], shape[1], shape[2]);
        let flat = normed2.reshape(&[batch * seq, feat]).unwrap();
        let h = self.mlp_fc1.forward(&flat);
        // GELU activation (approximate with ReLU for simplicity in autograd chain)
        let h = h.relu().unwrap();
        let h = self.mlp_fc2.forward(&h);
        let mlp_out = h.reshape(&[batch, seq, feat]).unwrap();

        x.add(&mlp_out).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.norm1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.mlp_fc1.parameters());
        p.extend(self.mlp_fc2.parameters());
        p
    }
}

// ---------------------------------------------------------------------------
// Vision Transformer (ViT)
// ---------------------------------------------------------------------------
pub struct ViT {
    pub patch_embed: PatchEmbedding,
    pub cls_token: Variable,
    pub pos_embed: Variable,
    pub blocks: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub head: Linear,
    pub num_patches: usize,
    pub embed_dim: usize,
}

impl ViT {
    pub fn new(
        img_channels: usize,
        img_size: usize,
        patch_size: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        num_classes: usize,
    ) -> Self {
        let patch_embed = PatchEmbedding::new(img_channels, img_size, patch_size, embed_dim);
        let num_patches = patch_embed.num_patches;

        // CLS token: [1, 1, embed_dim]
        let cls_data: Vec<f64> = {
            let mut rng = rand::thread_rng();
            (0..embed_dim).map(|_| rng.gen::<f64>() * 0.02).collect()
        };
        let cls_token = Variable::requires_grad(Tensor::from_slice(&cls_data, &[1, 1, embed_dim]));

        // Positional embedding: [1, num_patches + 1, embed_dim]
        let pos_data: Vec<f64> = {
            let mut rng = rand::thread_rng();
            (0..(num_patches + 1) * embed_dim)
                .map(|_| rng.gen::<f64>() * 0.02)
                .collect()
        };
        let pos_embed = Variable::requires_grad(Tensor::from_slice(
            &pos_data,
            &[1, num_patches + 1, embed_dim],
        ));

        let mlp_dim = embed_dim * 4;
        let blocks: Vec<TransformerBlock> = (0..num_blocks)
            .map(|_| TransformerBlock::new(embed_dim, num_heads, mlp_dim))
            .collect();

        let norm = LayerNorm::new(vec![embed_dim]);
        let head = Linear::new(embed_dim, num_classes);

        Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
            num_patches,
            embed_dim,
        }
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();

        // patch_embed
        sd.extend(self.patch_embed.state_dict_entries());

        // cls_token, pos_embed
        sd.insert("cls_token".to_string(), self.cls_token.tensor().clone());
        sd.insert("pos_embed".to_string(), self.pos_embed.tensor().clone());

        // blocks
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{i}.");
            sd.extend(block.state_dict_entries(&prefix));
        }

        // norm
        let norm_params = self.norm.parameters();
        sd.insert("norm.weight".to_string(), norm_params[0].tensor().clone());
        sd.insert("norm.bias".to_string(), norm_params[1].tensor().clone());

        // classifier
        let head_params = self.head.parameters();
        sd.insert("classifier.weight".to_string(), head_params[0].tensor().clone());
        sd.insert("classifier.bias".to_string(), head_params[1].tensor().clone());

        sd
    }

    pub fn from_state_dict(
        sd: &HashMap<String, Tensor>,
        img_channels: usize,
        img_size: usize,
        patch_size: usize,
        num_heads: usize,
        num_blocks: usize,
    ) -> Self {
        let patch_embed = PatchEmbedding::from_state_dict(sd, img_channels, img_size, patch_size);
        let num_patches = patch_embed.num_patches;

        let cls_token = Variable::requires_grad(sd["cls_token"].clone());
        let pos_embed = Variable::requires_grad(sd["pos_embed"].clone());

        let embed_dim = cls_token.tensor().shape()[2];

        let blocks: Vec<TransformerBlock> = (0..num_blocks)
            .map(|i| {
                let prefix = format!("blocks.{i}.");
                TransformerBlock::from_state_dict(sd, &prefix, num_heads)
            })
            .collect();

        let norm = LayerNorm::from_tensors(
            sd["norm.weight"].clone(),
            sd["norm.bias"].clone(),
        );

        let head = Linear::from_tensors(
            sd["classifier.weight"].clone(),
            Some(sd["classifier.bias"].clone()),
        );

        Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
            num_patches,
            embed_dim,
        }
    }
}

impl Module for ViT {
    fn forward(&self, input: &Variable) -> Variable {
        let batch_size = input.tensor().shape()[0];

        // 1. Patch embedding: [N, num_patches, embed_dim]
        let patches = self.patch_embed.forward(input);

        // 2. Prepend CLS token: expand cls_token to [N, 1, embed_dim]
        let cls_data = self.cls_token.tensor().to_vec_f64().unwrap();
        let mut expanded_cls = Vec::with_capacity(batch_size * self.embed_dim);
        for _ in 0..batch_size {
            expanded_cls.extend_from_slice(&cls_data);
        }
        let cls_expanded = Variable::new(Tensor::from_slice(
            &expanded_cls,
            &[batch_size, 1, self.embed_dim],
        ));

        // Concatenate CLS + patches along sequence dimension
        let patches_data = patches.tensor().to_vec_f64().unwrap();
        let cls_exp_data = cls_expanded.tensor().to_vec_f64().unwrap();
        let seq_len = self.num_patches + 1;
        let mut combined = vec![0.0f64; batch_size * seq_len * self.embed_dim];
        for b in 0..batch_size {
            // CLS token
            let cls_offset = b * self.embed_dim;
            let dst_offset = b * seq_len * self.embed_dim;
            combined[dst_offset..dst_offset + self.embed_dim]
                .copy_from_slice(&cls_exp_data[cls_offset..cls_offset + self.embed_dim]);
            // Patches
            let patch_offset = b * self.num_patches * self.embed_dim;
            let dst_patch = dst_offset + self.embed_dim;
            combined[dst_patch..dst_patch + self.num_patches * self.embed_dim]
                .copy_from_slice(
                    &patches_data[patch_offset..patch_offset + self.num_patches * self.embed_dim],
                );
        }
        let mut x = Variable::new(Tensor::from_slice(
            &combined,
            &[batch_size, seq_len, self.embed_dim],
        ));

        // 3. Add positional embedding (broadcast over batch)
        let pos_data = self.pos_embed.tensor().to_vec_f64().unwrap();
        let mut pos_expanded = Vec::with_capacity(batch_size * seq_len * self.embed_dim);
        for _ in 0..batch_size {
            pos_expanded.extend_from_slice(&pos_data);
        }
        let pos_var = Variable::new(Tensor::from_slice(
            &pos_expanded,
            &[batch_size, seq_len, self.embed_dim],
        ));
        x = x.add(&pos_var).unwrap();

        // 4. Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 5. Layer norm
        x = self.norm.forward(&x);

        // 6. Extract CLS token: x[:, 0, :]
        let x_data = x.tensor().to_vec_f64().unwrap();
        let mut cls_out = vec![0.0f64; batch_size * self.embed_dim];
        for b in 0..batch_size {
            let src = b * seq_len * self.embed_dim;
            let dst = b * self.embed_dim;
            cls_out[dst..dst + self.embed_dim]
                .copy_from_slice(&x_data[src..src + self.embed_dim]);
        }
        let cls_features =
            Variable::new(Tensor::from_slice(&cls_out, &[batch_size, self.embed_dim]));

        // 7. Classification head
        self.head.forward(&cls_features)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.patch_embed.parameters());
        p.push(self.cls_token.clone());
        p.push(self.pos_embed.clone());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.norm.parameters());
        p.extend(self.head.parameters());
        p
    }
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------
pub fn random_images(batch_size: usize, channels: usize, h: usize, w: usize) -> Tensor {
    let numel = batch_size * channels * h * w;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..numel).map(|_| rng.gen::<f64>()).collect();
    Tensor::from_slice(&data, &[batch_size, channels, h, w])
}

pub fn random_labels(batch_size: usize, num_classes: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..batch_size)
        .map(|_| rng.gen_range(0..num_classes) as f64)
        .collect();
    Tensor::from_slice(&data, &[batch_size])
}
