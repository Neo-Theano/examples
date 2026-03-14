# MNIST CNN

Convolutional neural network for MNIST digit classification using Neo Theano.

## Architecture

```
Conv2d(1, 32, kernel_size=3)    -> ReLU
Conv2d(32, 64, kernel_size=3)   -> ReLU -> MaxPool2d(2)
Dropout(0.25)
Flatten
Linear(9216, 128)               -> ReLU
Dropout(0.5)
Linear(128, 10)
```

**Total parameters:** 1,199,882

## Training

```bash
cargo run --release --bin mnist
```

**Hyperparameters:** Adam optimizer, lr=0.001, CrossEntropyLoss, 3 epochs, batch_size=4

### Training Output

```
Neo Theano — MNIST CNN Example
(Using synthetic random data)

Training...
  Epoch [1/3], Batch [1/10], Loss: 6.5781
  Epoch [1/3], Batch [6/10], Loss: 5.0536
  Epoch [1/3] Average Loss: 4.6780
  Epoch [2/3], Batch [1/10], Loss: 5.0535
  Epoch [2/3], Batch [6/10], Loss: 4.4328
  Epoch [2/3] Average Loss: 4.4095
  Epoch [3/3], Batch [1/10], Loss: 4.5461
  Epoch [3/3], Batch [6/10], Loss: 5.1148
  Epoch [3/3] Average Loss: 4.8247

Evaluating...
Test Accuracy: 5.00% (1/20 correct)
(Note: accuracy is ~10% random baseline since we use synthetic data)
Model saved to mnist_model.safetensors
```

## Inference

```bash
cargo run --release --bin mnist-infer
```

### Inference Output

```
=== MNIST CNN Inference ===

Model loaded from mnist_model.safetensors

--- Classifying synthetic images ---

  Sample 1:
    Top predictions:
      Digit 6: 0.2651 (26.5%)
      Digit 3: 0.2230 (22.3%)
      Digit 4: 0.1739 (17.4%)

  Sample 2:
    Top predictions:
      Digit 6: 0.2611 (26.1%)
      Digit 4: 0.2290 (22.9%)
      Digit 3: 0.2178 (21.8%)

  Sample 3:
    Top predictions:
      Digit 3: 0.2809 (28.1%)
      Digit 6: 0.2309 (23.1%)
      Digit 4: 0.1646 (16.5%)

Inference complete.
```

## Model

Saved to `mnist_model.safetensors` using SafeTensors format.
