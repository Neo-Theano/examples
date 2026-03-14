# SNLI (Natural Language Inference)

BiLSTM classifier for natural language inference — classifies premise-hypothesis pairs as entailment, contradiction, or neutral.

## Architecture

```
Premise:    Embedding(500, 64) -> LSTM(64, 64)
Hypothesis: Embedding(500, 64) -> LSTM(64, 64)
Combine:    concat(premise_hidden, hypothesis_hidden)
Classify:   Linear(128, 64) -> ReLU -> Linear(64, 3)
```

**Total parameters:** 11

## Training

```bash
cargo run --release --bin snli
```

**Hyperparameters:** Adam optimizer, lr=0.001, CrossEntropyLoss, 10 epochs, batch_size=16, seq_len=12

### Training Output

```
SNLI — Natural Language Inference Example

Vocab: 500, Embed: 64, Hidden: 64
Seq len: 12, Batch: 16, Classes: 3

Epoch [ 1/10] — Loss: 1.1408, Accuracy: 30.6%
Epoch [ 5/10] — Loss: 1.1401, Accuracy: 31.2%
Epoch [ 7/10] — Loss: 1.1005, Accuracy: 38.8%
Epoch [10/10] — Loss: 1.0931, Accuracy: 41.9%

Model saved to snli_model.safetensors
```

## Inference

```bash
cargo run --release --bin snli-infer
```

### Inference Output

```
=== SNLI Inference ===

Model loaded from snli_model.safetensors

--- Class probabilities ---
  Entailment:    0.3290 (logit: 0.0125)
  Contradiction: 0.3927 (logit: 0.1894)
  Neutral:       0.2783 (logit: -0.1547)

Prediction: Contradiction (confidence: 39.27%)

Inference complete.
```

## Model

Saved to `snli_model.safetensors` using SafeTensors format.
