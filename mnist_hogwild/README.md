# MNIST Hogwild

Demonstrates Hogwild!-style asynchronous SGD training with multiple simulated workers sharing model parameters.

## Architecture

Same MNIST CNN as the `mnist` example:

```
Conv2d(1, 32, 3) -> ReLU -> Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
-> Dropout(0.25) -> Flatten -> Linear(9216, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 10)
```

**Total parameters:** 1,199,882

## Training

```bash
cargo run --release --bin mnist_hogwild
```

**Hyperparameters:** 4 workers, 5 batches per worker, Adam optimizer, lr=0.001, 2 epochs

### Training Output

```
Neo Theano — MNIST Hogwild Example
(Demonstrating Hogwild-style shared-parameter training API)

Hogwild training with 4 processes
Each worker processes 5 batches of size 4 per epoch

Model: MNIST CNN (1199882 parameters)

--- Epoch 1/2 ---
  [Worker 0] Finished. Average Loss: 3.1012
  [Worker 1] Finished. Average Loss: 4.7083
  [Worker 2] Finished. Average Loss: 3.9518
  [Worker 3] Finished. Average Loss: 3.2154

--- Epoch 2/2 ---
  [Worker 0] Finished. Average Loss: 3.3234
  [Worker 1] Finished. Average Loss: 4.2101
  [Worker 2] Finished. Average Loss: 3.6804
  [Worker 3] Finished. Average Loss: 3.0758

Evaluating...
Test Accuracy: 5.00% (1/20 correct)
Model saved to mnist_hogwild_model.safetensors
```

## Inference

```bash
cargo run --release --bin mnist_hogwild-infer
```

### Inference Output

```
=== MNIST Hogwild CNN Inference ===

Model loaded from mnist_hogwild_model.safetensors

--- Classifying synthetic images ---

  Sample 1:
    Top predictions:
      Digit 7: 0.3265 (32.6%)
      Digit 2: 0.1878 (18.8%)
      Digit 9: 0.1145 (11.4%)

  Sample 2:
    Top predictions:
      Digit 7: 0.2982 (29.8%)
      Digit 3: 0.1891 (18.9%)
      Digit 2: 0.1465 (14.6%)

  Sample 3:
    Top predictions:
      Digit 7: 0.2669 (26.7%)
      Digit 2: 0.1873 (18.7%)
      Digit 3: 0.1476 (14.8%)

Inference complete.
```

## Model

Saved to `mnist_hogwild_model.safetensors` using SafeTensors format.
