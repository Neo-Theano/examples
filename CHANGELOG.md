# Changelog

## [0.2.0] - 2026-03-14

### Added

- **Model serialization** for all 20 trainable examples using SafeTensors format
  - Each example now saves trained weights to a `.safetensors` file after training
  - Models can be loaded back from saved files for inference
- **Inference binaries** (`src/bin/infer.rs`) for all 20 examples
  - Each loads a previously trained model and runs inference on sample data
  - Run with `cargo run --bin <name>-infer` (e.g., `cargo run --bin vae-infer`)
- **Library crates** (`src/lib.rs`) for all 20 examples
  - Model definitions extracted into reusable library modules
  - Each model exposes `state_dict()` and `from_state_dict()` for save/load
- **`from_tensors()` constructors** in `theano-nn` for reconstructing layers from loaded weights:
  - `Linear::from_tensors(weight, bias)`
  - `Conv2d::from_tensors(weight, bias, stride, padding)`
  - `Embedding::from_tensors(weight)`
  - `LSTMCell::from_tensors(w_ih, w_hh, b_ih, b_hh)`
  - `RNNCell::from_tensors(w_ih, w_hh, b_ih, b_hh)`
  - `LayerNorm::from_tensors(weight, bias)`

### Model output files

| Example | Train command | Infer command | Model file |
|---|---|---|---|
| mnist | `cargo run --bin mnist` | `cargo run --bin mnist-infer` | `mnist_model.safetensors` |
| mnist_rnn | `cargo run --bin mnist_rnn` | `cargo run --bin mnist_rnn-infer` | `mnist_rnn_model.safetensors` |
| mnist_forward_forward | `cargo run --bin mnist_forward_forward` | `cargo run --bin mnist_forward_forward-infer` | `mnist_ff_model.safetensors` |
| mnist_hogwild | `cargo run --bin mnist_hogwild` | `cargo run --bin mnist_hogwild-infer` | `mnist_hogwild_model.safetensors` |
| regression | `cargo run --bin regression` | `cargo run --bin regression-infer` | `regression_model.safetensors` |
| reinforcement_learning | `cargo run --bin reinforcement_learning` | `cargo run --bin reinforcement_learning-infer` | `rl_model.safetensors` |
| vae | `cargo run --bin vae` | `cargo run --bin vae-infer` | `vae_model.safetensors` |
| dcgan | `cargo run --bin dcgan` | `cargo run --bin dcgan-infer` | `dcgan_model.safetensors` |
| super_resolution | `cargo run --bin super_resolution` | `cargo run --bin super_resolution-infer` | `super_resolution_model.safetensors` |
| time_sequence_prediction | `cargo run --bin time_sequence_prediction` | `cargo run --bin time_sequence_prediction-infer` | `time_seq_model.safetensors` |
| distributed | `cargo run --bin distributed` | `cargo run --bin distributed-infer` | `distributed_model.safetensors` |
| fast_neural_style | `cargo run --bin fast_neural_style` | `cargo run --bin fast_neural_style-infer` | `style_model.safetensors` |
| siamese_network | `cargo run --bin siamese_network` | `cargo run --bin siamese_network-infer` | `siamese_model.safetensors` |
| language_translation | `cargo run --bin language_translation` | `cargo run --bin language_translation-infer` | `translation_model.safetensors` |
| snli | `cargo run --bin snli` | `cargo run --bin snli-infer` | `snli_model.safetensors` |
| gcn | `cargo run --bin gcn` | `cargo run --bin gcn-infer` | `gcn_model.safetensors` |
| gat | `cargo run --bin gat` | `cargo run --bin gat-infer` | `gat_model.safetensors` |
| imagenet | `cargo run --bin imagenet` | `cargo run --bin imagenet-infer` | `resnet18_model.safetensors` |
| vision_transformer | `cargo run --bin vision_transformer` | `cargo run --bin vision_transformer-infer` | `vit_model.safetensors` |
| word_language_model | `cargo run --bin word_language_model` | `cargo run --bin word_language_model-infer` | `word_lm_model.safetensors` |

## [0.1.0] - 2026-03-13

### Added

- Initial implementation of all 22 PyTorch examples in Neo Theano (Rust)
