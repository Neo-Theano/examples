# MNIST RNN

LSTM-based recurrent neural network for MNIST digit classification, treating each 28x28 image as a sequence of 28 timesteps with 28-dimensional input vectors.

## Architecture

```
Input: 28x28 image -> 28 timesteps of 28-dim vectors
LSTMCell(input_size=28, hidden_size=128)
Linear(128, 10)
```

**Total parameters:** 82,186

## Training

```bash
cargo run --release --bin mnist_rnn
```

**Hyperparameters:** Adam optimizer, lr=0.001, CrossEntropyLoss, 3 epochs

### Training Output

```
Neo Theano — MNIST RNN Example
(Using LSTM to classify MNIST digits as sequences)

Training...
  Epoch [1/3], Batch [1/10], Loss: 2.5823
  Epoch [1/3] Average Loss: 2.5093
  Epoch [2/3] Average Loss: 2.4048
  Epoch [3/3] Average Loss: 2.2797

Evaluating...
Test Accuracy: 10.00% (2/20 correct)
Model saved to mnist_rnn_model.safetensors
```

## Inference

```bash
cargo run --release --bin mnist_rnn-infer
```

### Inference Output

```
Model loaded from mnist_rnn_model.safetensors
  LSTM hidden_size=128, seq_len=28, input_size=28

--- Classifying synthetic images (row-by-row as sequences) ---

  Sample 1:
    Top predictions:
      Digit 5: 0.1446 (14.5%)
      Digit 3: 0.1420 (14.2%)
      Digit 8: 0.1161 (11.6%)

  Sample 2:
    Top predictions:
      Digit 5: 0.1800 (18.0%)
      Digit 1: 0.1344 (13.4%)
      Digit 8: 0.1089 (10.9%)

  Sample 3:
    Top predictions:
      Digit 5: 0.1743 (17.4%)
      Digit 1: 0.1509 (15.1%)
      Digit 8: 0.1337 (13.4%)

Inference complete.
```

## Model

Saved to `mnist_rnn_model.safetensors` using SafeTensors format.
