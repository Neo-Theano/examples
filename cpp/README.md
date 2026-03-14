# C++ Frontend API Demo (Rust-Native)

Comprehensive showcase of the Neo Theano Rust-native API, equivalent to PyTorch's C++ frontend (`libtorch`).

## Features Demonstrated

1. **Tensor creation**: zeros, ones, full, arange, linspace, eye, from_slice, scalar
2. **Tensor operations**: arithmetic (+, -, *, /), matmul, sum, mean, max, min, relu, exp
3. **Tensor views**: reshape, transpose, unsqueeze, flatten
4. **Autograd**: requires_grad, backward, gradient computation
5. **Neural network layers**: Linear, Conv2d, MaxPool2d, AdaptiveAvgPool2d, BatchNorm1d, LayerNorm, Embedding, Flatten
6. **Sequential model**: composing multiple layers
7. **Loss functions**: MSELoss, CrossEntropyLoss
8. **Optimizers**: SGD, Adam, AdamW
9. **End-to-end training loop**: 10 epochs on synthetic data
10. **Inference**: with NoGradGuard, softmax probability predictions

## Running

```bash
cargo run --release --bin cpp
```

### Output

```
--- Tensor Creation ---
zeros([2,3]):   shape=[2, 3]
ones([3,2]):    shape=[3, 2]
full([2,2], pi): [3.14, 3.14, 3.14, 3.14]
arange(0,5,1):  [0.0, 1.0, 2.0, 3.0, 4.0]
linspace(0,1,5): [0.0, 0.25, 0.5, 0.75, 1.0]
eye(3):         [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

--- Autograd ---
x = [2, 3]
y = x^2 = [4.0, 9.0]
loss = sum(y) = 13
grad = d(loss)/dx = [4.0, 6.0] (expected [4, 6])

--- Neural Network Layers ---
Linear(10, 5):  input=[4, 10] -> output=[4, 5]
Conv2d(3, 16, 3x3): input=[1, 3, 8, 8] -> output=[1, 16, 8, 8]
MaxPool2d(2):   input=[1, 16, 8, 8] -> output=[1, 16, 4, 4]

--- Sequential Model ---
MLP model: 784 -> 256 -> 128 -> 10
Parameters: 235146

--- End-to-End Training ---
  Epoch [ 5/10]  Loss: 1.3704
  Epoch [10/10]  Loss: 1.7847

--- Inference ---
  Sample 0: probs=[0.251, 0.592, 0.157] -> class 1
  Sample 1: probs=[0.150, 0.685, 0.165] -> class 1

Theano Rust-native API demo complete.
```

## Notes

This is a demonstration-only example. No model is saved — it showcases the full API surface of Neo Theano.
