# Word Language Model

LSTM-based language model that learns to predict the next token in a sequence, reporting perplexity.

## Architecture

```
Embedding(vocab_size=1000, embed_dim=64)
LSTMCell(input_size=64, hidden_size=128)
Linear(128, 1000)
```

**Total parameters:** 7

## Training

```bash
cargo run --release --bin word_language_model
```

**Hyperparameters:** SGD optimizer, lr=0.01, CrossEntropyLoss, 5 epochs, seq_len=35, batch_size=16

### Training Output

```
Word Language Model -- LSTM Language Model Example

Vocab size: 1000, Embed dim: 64, Hidden: 128
Seq length: 35, Batch size: 16

Epoch [1/5] -- Loss: 6.9625, Perplexity: 1056.25
Epoch [2/5] -- Loss: 6.9617, Perplexity: 1055.39
Epoch [3/5] -- Loss: 6.9665, Perplexity: 1060.53
Epoch [4/5] -- Loss: 6.9587, Perplexity: 1052.23
Epoch [5/5] -- Loss: 6.9692, Perplexity: 1063.40

Model saved to word_lm_model.safetensors (2339198 bytes)
```

## Inference

```bash
cargo run --release --bin word_language_model-infer
```

### Inference Output

```
=== Word Language Model (LSTM) Inference ===

Loaded state dict with 7 tensors
LSTM language model reconstructed.
  Vocab size:   1000
  Hidden size:  128

Sequence perplexity: 971.00

Top-5 next-token predictions (last timestep):
  #1: token 972 (score: 1.1560)
  #2: token 227 (score: 1.1308)
  #3: token 232 (score: 1.1029)
  #4: token 165 (score: 0.9460)
  #5: token 6 (score: 0.9377)

Inference complete.
```

## Model

Saved to `word_lm_model.safetensors` using SafeTensors format.
