# FX Graph Mode / JIT

Demonstrates theano-jit graph construction, optimization passes, and IR operations.

## Features

- Manual graph building with Constant, MatMul, ReLU, Sigmoid, Mean operations
- Dead code elimination optimization pass
- Neural network graph IR construction
- Graph tracing with `trace()`
- Full IR operations catalogue

## Running

```bash
cargo run --release --bin fx
```

### Output

```
--- Graph Construction ---

Graph BEFORE optimisation (9 nodes):
  %0 = constant([2, 2])
  %1 = constant([2, 2])
  %2 = matmul(%0, %1)
  %3 = relu(%2)
  %4 = sigmoid(%2)   // dead node
  ...

Graph AFTER optimisation (5 nodes):
  %0 = constant([2, 2])
  %1 = constant([2, 2])
  %2 = matmul(%0, %1)
  %3 = relu(%2)
  %4 = mean(%3)

Removed 4 dead nodes.

--- Neural Network Graph ---
  %0 = constant([2, 4])  // input
  %1 = constant([4, 3])  // weights
  %4 = matmul(%0, %1)
  %5 = add(%4, %2)       // + bias
  %6 = relu(%5)
  %7 = matmul(%6, %3)
  %8 = mean(%7)

--- IR Operations Catalogue ---
  add, sub, mul, div, neg, exp, log, sqrt,
  tanh, sigmoid, relu, sum, mean, reshape, transpose

FX graph mode demonstration complete.
```

## Notes

This is a demonstration-only example. No model is saved or loaded — it showcases the graph IR and optimization capabilities of `theano-jit`.
