# Neo Theano Examples

A collection of examples demonstrating the [Neo Theano](https://github.com/Neo-Theano/theano) deep learning framework in Rust. These examples mirror the official [PyTorch Examples](https://github.com/pytorch/examples) repository, showing equivalent implementations in Rust.

## Examples

### Vision

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [mnist](mnist/) | CNN for MNIST digit classification | [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/main/mnist) |
| [mnist_hogwild](mnist_hogwild/) | Multi-threaded MNIST training (Hogwild) | [pytorch/examples/mnist_hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild) |
| [mnist_rnn](mnist_rnn/) | RNN/LSTM for MNIST (rows as sequences) | [pytorch/examples/mnist_rnn](https://github.com/pytorch/examples/tree/main/mnist_rnn) |
| [mnist_forward_forward](mnist_forward_forward/) | Forward-Forward algorithm (Hinton 2022) | [pytorch/examples/mnist_forward_forward](https://github.com/pytorch/examples/tree/main/mnist_forward_forward) |
| [imagenet](imagenet/) | ResNet-18 for ImageNet classification | [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/main/imagenet) |
| [super_resolution](super_resolution/) | Sub-pixel CNN for image upscaling | [pytorch/examples/super_resolution](https://github.com/pytorch/examples/tree/main/super_resolution) |
| [fast_neural_style](fast_neural_style/) | Neural style transfer | [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/main/fast_neural_style) |
| [siamese_network](siamese_network/) | Siamese network for similarity learning | [pytorch/examples/siamese_network](https://github.com/pytorch/examples/tree/main/siamese_network) |
| [vision_transformer](vision_transformer/) | Vision Transformer (ViT) | - |

### Generative Models

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [vae](vae/) | Variational Autoencoder | [pytorch/examples/vae](https://github.com/pytorch/examples/tree/main/vae) |
| [dcgan](dcgan/) | Deep Convolutional GAN | [pytorch/examples/dcgan](https://github.com/pytorch/examples/tree/main/dcgan) |

### NLP / Sequence Modeling

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [word_language_model](word_language_model/) | RNN/LSTM language model | [pytorch/examples/word_language_model](https://github.com/pytorch/examples/tree/main/word_language_model) |
| [language_translation](language_translation/) | Transformer seq2seq translation | [pytorch/examples/language_translation](https://github.com/pytorch/examples/tree/main/language_translation) |
| [snli](snli/) | Natural language inference | [pytorch/examples/snli](https://github.com/pytorch/examples/tree/main/snli) |
| [time_sequence_prediction](time_sequence_prediction/) | LSTM for time series (sine waves) | [pytorch/examples/time_sequence_prediction](https://github.com/pytorch/examples/tree/main/time_sequence_prediction) |

### Reinforcement Learning

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [reinforcement_learning](reinforcement_learning/) | REINFORCE policy gradient | [pytorch/examples/reinforcement_learning](https://github.com/pytorch/examples/tree/main/reinforcement_learning) |

### Graph Neural Networks

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [gcn](gcn/) | Graph Convolutional Network | [pytorch/examples/gcn](https://github.com/pytorch/examples/tree/main/gcn) |
| [gat](gat/) | Graph Attention Network | [pytorch/examples/gat](https://github.com/pytorch/examples/tree/main/gat) |

### Other

| Example | Description | PyTorch Equivalent |
|---|---|---|
| [regression](regression/) | Polynomial regression | [pytorch/examples/regression](https://github.com/pytorch/examples/tree/main/regression) |
| [distributed](distributed/) | Distributed Data Parallel training | [pytorch/examples/distributed](https://github.com/pytorch/examples/tree/main/distributed) |
| [fx](fx/) | JIT graph capture and optimization | [pytorch/examples/fx](https://github.com/pytorch/examples/tree/main/fx) |
| [cpp](cpp/) | Rust-native API usage (C++ frontend equivalent) | [pytorch/examples/cpp](https://github.com/pytorch/examples/tree/main/cpp) |

## Prerequisites

- Rust 1.75+
- Neo Theano framework (cloned alongside this repo)

Expected directory layout:
```
theano/
  rusttorch/        # Neo Theano framework
  theano-examples/  # This repository
```

## Running Examples

```bash
# Build all examples
cargo build --workspace

# Run a specific example
cargo run -p mnist
cargo run -p vae
cargo run -p imagenet
cargo run -p vision_transformer

# Run with release optimizations (recommended for larger models)
cargo run -p imagenet --release
```

## Notes

- Examples use **synthetic/random data** to demonstrate the API without requiring dataset downloads. The model architectures, training loops, and loss computations are fully functional.
- To train on real datasets, replace the synthetic data generation with actual dataset loading (MNIST, CIFAR-10, ImageNet, etc.).
- GPU backends (CUDA, ROCm, Metal) can be enabled via feature flags in the theano dependency.

## License

MIT OR Apache-2.0 (same as Neo Theano)
