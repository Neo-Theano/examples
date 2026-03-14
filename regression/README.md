# Polynomial Regression

Fits a polynomial y = a*x^3 + b*x^2 + c*x + d using a linear model with polynomial features.

## Architecture

```
Linear(3, 1)  — maps polynomial features [x, x^2, x^3] to y
(bias term acts as the constant d)
```

**Total parameters:** 4

## Training

```bash
cargo run --release --bin regression
```

**Hyperparameters:** SGD optimizer, lr=0.01, MSELoss, 200 epochs, 200 training samples

### Training Output

```
Neo Theano — Polynomial Regression Example
Fitting: y = a*x^3 + b*x^2 + c*x + d
True coefficients: a=0.5, b=-1.2, c=2, d=-0.3

Training...
  Epoch [  1/200], Loss: 436.071316
  Epoch [101/200], Loss: 436.071316
  Epoch [200/200], Loss: 436.071316

=== Learned Coefficients ===
  a (x^3): 2.9253  (true: 0.5)
  b (x^2): -1.1847  (true: -1.2)
  c (x):   0.6793  (true: 2)
  d:       0.0272  (true: -0.3)

Test MSE Loss: 390.779745
Model saved to regression_model.safetensors
```

## Inference

```bash
cargo run --release --bin regression-infer
```

### Inference Output

```
=== Polynomial Regression Inference ===

Model loaded from regression_model.safetensors

--- Learned Coefficients ---
  a (x^3): -1.2496  (true: 0.5)
  b (x^2): -1.0470  (true: -1.2)
  c (x):   -0.0037  (true: 2)
  d:       0.0450  (true: -0.3)

--- Predictions ---
  x =  -2.0  ->  predicted:  5.8614,  true: -13.1000
  x =  -1.0  ->  predicted:  0.2513,  true: -4.0000
  x =   0.0  ->  predicted:  0.0450,  true: -0.3000
  x =   1.0  ->  predicted: -2.2553,  true:  1.0000
  x =   2.0  ->  predicted: -14.1471,  true:  2.9000

Inference complete.
```

## Model

Saved to `regression_model.safetensors` using SafeTensors format.
