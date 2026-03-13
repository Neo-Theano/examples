//! Rust-Native API Example (equivalent of PyTorch C++ Frontend)
//!
//! Demonstrates the full public API surface of the Theano framework used as a
//! native Rust library: tensor creation, operations, model building, training,
//! and inference.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{
    Conv2d, Linear, ReLU, Softmax, Sequential,
    BatchNorm1d, LayerNorm, Embedding, MaxPool2d, AdaptiveAvgPool2d, Flatten,
    MSELoss, CrossEntropyLoss, Module,
};
use theano_optim::{SGD, Adam, AdamW, Optimizer};

fn main() {
    println!("=== Theano Rust-Native API Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Tensor creation
    // -----------------------------------------------------------------------
    println!("--- Tensor Creation ---\n");

    let zeros = Tensor::zeros(&[2, 3]);
    println!("zeros([2,3]):   shape={:?}", zeros.shape());

    let ones = Tensor::ones(&[3, 2]);
    println!("ones([3,2]):    shape={:?}", ones.shape());

    let full = Tensor::full(&[2, 2], 3.14);
    println!("full([2,2], pi): {:?}", full.to_vec_f64().unwrap());

    let arange = Tensor::arange(0.0, 5.0, 1.0);
    println!("arange(0,5,1):  {:?}", arange.to_vec_f64().unwrap());

    let linspace = Tensor::linspace(0.0, 1.0, 5);
    println!("linspace(0,1,5): {:?}", linspace.to_vec_f64().unwrap());

    let eye = Tensor::eye(3);
    println!("eye(3):         {:?}", eye.to_vec_f64().unwrap());

    let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    println!("from_slice:     shape={:?} data={:?}", from_data.shape(), from_data.to_vec_f64().unwrap());

    let scalar = Tensor::scalar(42.0);
    println!("scalar(42):     item={}", scalar.item().unwrap());

    // -----------------------------------------------------------------------
    // 2. Tensor operations
    // -----------------------------------------------------------------------
    println!("\n--- Tensor Operations ---\n");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    println!("a + b = {:?}", (&a + &b).to_vec_f64().unwrap());
    println!("a - b = {:?}", (&a - &b).to_vec_f64().unwrap());
    println!("a * b = {:?}", (&a * &b).to_vec_f64().unwrap());
    println!("a / b = {:?}", (&a / &b).to_vec_f64().unwrap());

    let c = a.matmul(&b).unwrap();
    println!("a @ b = {:?}  shape={:?}", c.to_vec_f64().unwrap(), c.shape());

    println!("sum(a)  = {}", a.sum().unwrap().item().unwrap());
    println!("mean(a) = {}", a.mean().unwrap().item().unwrap());
    println!("max(a)  = {}", a.max().unwrap().item().unwrap());
    println!("min(a)  = {}", a.min().unwrap().item().unwrap());

    let relu_result = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]).relu().unwrap();
    println!("relu([-1,0,1,2]) = {:?}", relu_result.to_vec_f64().unwrap());

    let exp_result = Tensor::from_slice(&[0.0, 1.0], &[2]).exp().unwrap();
    println!("exp([0,1]) = {:?}", exp_result.to_vec_f64().unwrap());

    // -----------------------------------------------------------------------
    // 3. Tensor views and reshaping
    // -----------------------------------------------------------------------
    println!("\n--- Tensor Views ---\n");

    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("original:  shape={:?}", t.shape());

    let reshaped = t.reshape(&[3, 2]).unwrap();
    println!("reshape:   shape={:?}", reshaped.shape());

    let transposed = t.transpose(0, 1).unwrap();
    println!("transpose: shape={:?}", transposed.shape());

    let unsqueezed = t.unsqueeze(0).unwrap();
    println!("unsqueeze: shape={:?}", unsqueezed.shape());

    let flat = t.flatten(0, 1).unwrap();
    println!("flatten:   shape={:?}", flat.shape());

    // -----------------------------------------------------------------------
    // 4. Autograd
    // -----------------------------------------------------------------------
    println!("\n--- Autograd ---\n");

    let x = Variable::requires_grad(Tensor::from_slice(&[2.0, 3.0], &[2]));
    let y = x.mul(&x).unwrap(); // y = x^2
    let loss = y.sum().unwrap(); // loss = sum(x^2) = 4 + 9 = 13
    println!("x = [2, 3]");
    println!("y = x^2 = {:?}", y.tensor().to_vec_f64().unwrap());
    println!("loss = sum(y) = {}", loss.tensor().item().unwrap());

    loss.backward();
    let grad = x.grad().unwrap();
    println!("grad = d(loss)/dx = {:?} (expected [4, 6])", grad.to_vec_f64().unwrap());

    // -----------------------------------------------------------------------
    // 5. Neural network layers
    // -----------------------------------------------------------------------
    println!("\n--- Neural Network Layers ---\n");

    let linear = Linear::new(10, 5);
    let input = Variable::new(Tensor::ones(&[4, 10]));
    let out = linear.forward(&input);
    println!("Linear(10, 5):  input={:?} -> output={:?}", input.tensor().shape(), out.tensor().shape());

    let conv = Conv2d::with_options(3, 16, (3, 3), (1, 1), (1, 1), true);
    let img = Variable::new(Tensor::ones(&[1, 3, 8, 8]));
    let conv_out = conv.forward(&img);
    println!("Conv2d(3, 16, 3x3): input={:?} -> output={:?}", img.tensor().shape(), conv_out.tensor().shape());

    let pool = MaxPool2d::new(2);
    let pool_out = pool.forward(&conv_out);
    println!("MaxPool2d(2):   input={:?} -> output={:?}", conv_out.tensor().shape(), pool_out.tensor().shape());

    let ada_pool = AdaptiveAvgPool2d::new((1, 1));
    let ada_out = ada_pool.forward(&conv_out);
    println!("AdaptiveAvgPool2d(1,1): input={:?} -> output={:?}", conv_out.tensor().shape(), ada_out.tensor().shape());

    let bn = BatchNorm1d::new(5);
    let bn_input = Variable::new(Tensor::ones(&[4, 5]));
    let bn_out = bn.forward(&bn_input);
    println!("BatchNorm1d(5): input={:?} -> output={:?}", bn_input.tensor().shape(), bn_out.tensor().shape());

    let ln = LayerNorm::new(vec![5]);
    let ln_out = ln.forward(&bn_input);
    println!("LayerNorm([5]): input={:?} -> output={:?}", bn_input.tensor().shape(), ln_out.tensor().shape());

    let emb = Embedding::new(100, 16);
    let indices = Variable::new(Tensor::from_slice(&[0.0, 5.0, 10.0, 50.0], &[2, 2]));
    let emb_out = emb.forward(&indices);
    println!("Embedding(100, 16): input={:?} -> output={:?}", indices.tensor().shape(), emb_out.tensor().shape());

    let flatten_layer = Flatten::new();
    let flat_input = Variable::new(Tensor::ones(&[2, 3, 4, 5]));
    let flat_out = flatten_layer.forward(&flat_input);
    println!("Flatten:        input={:?} -> output={:?}", flat_input.tensor().shape(), flat_out.tensor().shape());

    // -----------------------------------------------------------------------
    // 6. Sequential model
    // -----------------------------------------------------------------------
    println!("\n--- Sequential Model ---\n");

    let model = Sequential::new(vec![])
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));

    let num_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("MLP model: 784 -> 256 -> 128 -> 10");
    println!("Parameters: {}", num_params);

    let batch = Variable::new(Tensor::ones(&[8, 784]));
    let logits = model.forward(&batch);
    println!("Forward:    input={:?} -> output={:?}", batch.tensor().shape(), logits.tensor().shape());

    // -----------------------------------------------------------------------
    // 7. Loss functions
    // -----------------------------------------------------------------------
    println!("\n--- Loss Functions ---\n");

    let pred = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]));
    let target_mse = Variable::new(Tensor::from_slice(&[1.5, 2.5, 3.5], &[1, 3]));
    let mse = MSELoss::new().forward(&pred, &target_mse);
    println!("MSELoss:          {:.4}", mse.tensor().item().unwrap());

    let logits_ce = Variable::new(Tensor::from_slice(&[2.0, 1.0, 0.1], &[1, 3]));
    let target_ce = Variable::new(Tensor::from_slice(&[0.0], &[1]));
    let ce = CrossEntropyLoss::new().forward(&logits_ce, &target_ce);
    println!("CrossEntropyLoss: {:.4}", ce.tensor().item().unwrap());

    // -----------------------------------------------------------------------
    // 8. Optimisers
    // -----------------------------------------------------------------------
    println!("\n--- Optimisers ---\n");

    let param = Variable::requires_grad(Tensor::from_slice(&[5.0, 5.0], &[2]));
    let mut sgd = SGD::new(vec![param], 0.1).momentum(0.9);
    sgd.params[0].tensor().set_grad(Tensor::from_slice(&[1.0, 1.0], &[2]));
    sgd.step();
    println!("SGD step: {:?}", sgd.params()[0].tensor().to_vec_f64().unwrap());

    let param2 = Variable::requires_grad(Tensor::from_slice(&[5.0, 5.0], &[2]));
    let mut adam = Adam::new(vec![param2], 0.01);
    adam.params[0].tensor().set_grad(Tensor::from_slice(&[1.0, 1.0], &[2]));
    adam.step();
    println!("Adam step: {:?}", adam.params()[0].tensor().to_vec_f64().unwrap());

    let param3 = Variable::requires_grad(Tensor::from_slice(&[5.0, 5.0], &[2]));
    let mut adamw = AdamW::new(vec![param3], 0.01).weight_decay(0.01);
    adamw.params[0].tensor().set_grad(Tensor::from_slice(&[1.0, 1.0], &[2]));
    adamw.step();
    println!("AdamW step: {:?}", adamw.params()[0].tensor().to_vec_f64().unwrap());

    // -----------------------------------------------------------------------
    // 9. End-to-end training loop
    // -----------------------------------------------------------------------
    println!("\n--- End-to-End Training ---\n");

    let train_model = Sequential::new(vec![])
        .add(Linear::new(4, 16))
        .add(ReLU)
        .add(Linear::new(16, 3));

    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(train_model.parameters(), 0.01);

    let mut rng = rand::thread_rng();
    for epoch in 0..10 {
        let data_vec: Vec<f64> = (0..8 * 4).map(|_| rng.gen::<f64>()).collect();
        let label_vec: Vec<f64> = (0..8).map(|_| rng.gen_range(0..3) as f64).collect();

        let data = Variable::new(Tensor::from_slice(&data_vec, &[8, 4]));
        let labels = Variable::new(Tensor::from_slice(&label_vec, &[8]));

        optimizer.zero_grad();
        let output = train_model.forward(&data);
        let loss = criterion.forward(&output, &labels);
        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        if (epoch + 1) % 5 == 0 {
            println!("  Epoch [{:>2}/10]  Loss: {:.4}", epoch + 1, loss_val);
        }
    }

    // -----------------------------------------------------------------------
    // 10. Inference (no_grad)
    // -----------------------------------------------------------------------
    println!("\n--- Inference ---\n");

    {
        let _guard = theano::autograd::NoGradGuard::new();
        let test_data_vec: Vec<f64> = (0..2 * 4).map(|_| rng.gen::<f64>()).collect();
        let test_input = Variable::new(Tensor::from_slice(&test_data_vec, &[2, 4]));
        let predictions = train_model.forward(&test_input);
        let sm = Softmax::new(-1);
        let probs = sm.forward(&predictions);
        println!("Test input shape:  {:?}", test_input.tensor().shape());
        println!("Predictions shape: {:?}", probs.tensor().shape());
        let probs_data = probs.tensor().to_vec_f64().unwrap();
        for i in 0..2 {
            let sample_probs = &probs_data[i * 3..(i + 1) * 3];
            let max_class = sample_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            println!(
                "  Sample {}: probs=[{:.3}, {:.3}, {:.3}] -> class {}",
                i, sample_probs[0], sample_probs[1], sample_probs[2], max_class
            );
        }
    }

    println!("\nTheano Rust-native API demo complete.");
}
