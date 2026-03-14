# Language Translation

Encoder-decoder Transformer for machine translation with greedy decoding.

## Architecture

```
Encoder: Embedding(500, 64) + Positional Encoding -> TransformerEncoder
Decoder: Embedding(500, 64) + Positional Encoding -> TransformerDecoder
Output:  Linear(64, 500)
```

**Total parameters:** 30

## Training

```bash
cargo run --release --bin language_translation
```

**Hyperparameters:** Adam optimizer, lr=0.001, CrossEntropyLoss, 5 epochs, src/tgt_vocab=500, seq_len=10, batch_size=8

### Training Output

```
Language Translation — Transformer Seq2Seq Example

Src vocab: 500, Tgt vocab: 500, Embed dim: 64
Src len: 10, Tgt len: 10, Batch: 8

Epoch [1/5] — Loss: 19.2711
Epoch [2/5] — Loss: 19.7699
Epoch [3/5] — Loss: 20.8162
Epoch [4/5] — Loss: 19.6362
Epoch [5/5] — Loss: 20.0957

Sample translation (greedy decode):
  Input tokens:     [223, 315, 238, 452, 112, 495, 45, 132, 57, 248]
  Predicted tokens: [251, 222, 178, 400, 196, 57, 222, 428, 196, 493]

Model saved to translation_model.safetensors
```

## Inference

```bash
cargo run --release --bin language_translation-infer
```

### Inference Output

```
=== Language Translation Inference ===

Model loaded from translation_model.safetensors

--- Translating synthetic source sequence ---
  Source tokens:    [75, 111, 297, 226, 222, 411, 301, 273, 454, 153]
  Predicted tokens: [220, 251, 294, 475, 193, 227, 220, 220, 294, 475]

--- Output logits (first 3 positions, top-5 tokens) ---
  Position 0: [220:15.89, 104:12.06, 362:11.62, 213:11.18, 494:9.96]
  Position 1: [251:15.86, 263:13.26, 8:13.16, 20:12.48, 29:12.09]
  Position 2: [294:19.40, 218:16.08, 243:15.82, 75:15.63, 210:13.29]

Inference complete.
```

## Model

Saved to `translation_model.safetensors` using SafeTensors format.
