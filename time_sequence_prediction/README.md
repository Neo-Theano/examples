# Time Sequence Prediction

LSTM network for sine wave prediction with teacher forcing during training and free-running prediction at inference.

## Architecture

```
LSTMCell(input_size=1, hidden_size=32)
Linear(32, 1)
```

**Total parameters:** 6

## Training

```bash
cargo run --release --bin time_sequence_prediction
```

**Hyperparameters:** Adam optimizer, lr=0.001, MSELoss, 10 epochs, seq_len=50, 20 sequences

### Training Output

```
Time Sequence Prediction -- LSTM Sine Wave Example

Hidden: 32, Seq len: 50, Sequences: 20

Epoch [ 1/10] -- MSE Loss: 0.614033
Epoch [ 5/10] -- MSE Loss: 0.708004
Epoch [10/10] -- MSE Loss: 0.688777

Predicted vs Actual (last test sequence, free-running):
Step      Predicted     Actual
1            0.0558     0.5646
2            0.0849     0.6442
3            0.1198     0.7174
...

Model saved to time_seq_model.safetensors
```

## Inference

```bash
cargo run --release --bin time_sequence_prediction-infer
```

### Inference Output

```
=== Time Sequence Prediction Inference ===

Model loaded from time_seq_model.safetensors

Predicted vs Actual (free-running prediction):
Step      Predicted     Actual      Error
1            0.0558     0.5646     0.5088
2            0.0849     0.6442     0.5593
3            0.1198     0.7174     0.5975
...

Overall MSE: 0.625404

Inference complete.
```

## Model

Saved to `time_seq_model.safetensors` using SafeTensors format.
