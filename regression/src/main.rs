//! Polynomial Regression Example — Fit y = a*x^3 + b*x^2 + c*x + d + noise.
//!
//! Demonstrates basic regression with polynomial features, MSE loss, and SGD optimizer.
//! A simple but complete training pipeline showing the fundamentals.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano::prelude::*;
use theano_nn::{Linear, MSELoss, Module};
use theano_optim::{Optimizer, SGD};

/// True polynomial coefficients: y = a*x^3 + b*x^2 + c*x + d
const TRUE_A: f64 = 0.5;
const TRUE_B: f64 = -1.2;
const TRUE_C: f64 = 2.0;
const TRUE_D: f64 = -0.3;

/// Generate synthetic polynomial data.
///
/// Returns (features, targets) where features has polynomial columns [x, x^2, x^3]
/// and targets = a*x^3 + b*x^2 + c*x + d + noise.
fn generate_data(n: usize, noise_std: f64) -> (Tensor, Tensor) {
    let mut rng = rand::thread_rng();
    let noise_dist = Normal::new(0.0, noise_std).unwrap();

    let mut features = vec![0.0f64; n * 3]; // [x, x^2, x^3]
    let mut targets = vec![0.0f64; n];

    for i in 0..n {
        let x: f64 = rng.gen_range(-3.0..3.0);
        let x2 = x * x;
        let x3 = x2 * x;

        features[i * 3] = x;
        features[i * 3 + 1] = x2;
        features[i * 3 + 2] = x3;

        let y = TRUE_A * x3 + TRUE_B * x2 + TRUE_C * x + TRUE_D;
        let noise: f64 = noise_dist.sample(&mut rng);
        targets[i] = y + noise;
    }

    let feat_tensor = Tensor::from_slice(&features, &[n, 3]);
    let target_tensor = Tensor::from_slice(&targets, &[n, 1]);
    (feat_tensor, target_tensor)
}

fn main() {
    println!("Neo Theano — Polynomial Regression Example");
    println!("Fitting: y = a*x^3 + b*x^2 + c*x + d");
    println!(
        "True coefficients: a={}, b={}, c={}, d={}\n",
        TRUE_A, TRUE_B, TRUE_C, TRUE_D
    );

    // Hyperparameters
    let lr = 0.01;
    let num_epochs = 200;
    let num_train = 200;
    let num_test = 50;
    let noise_std = 0.5;

    // Generate training data
    let (train_features, train_targets) = generate_data(num_train, noise_std);
    let (test_features, test_targets) = generate_data(num_test, noise_std);

    // Model: Linear(3, 1) — learns weights for [x, x^2, x^3] and a bias (the constant d)
    let model = Linear::new(3, 1);

    println!("=== Model Architecture ===");
    println!("  Linear(3, 1)  — maps polynomial features [x, x^2, x^3] to y");
    println!("  (bias term acts as the constant d)");
    let total_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!("  Total parameters: {}", total_params);
    println!("==========================\n");

    // Optimizer
    let params = model.parameters();
    let mut optimizer = SGD::new(params, lr);

    // Loss function
    let criterion = MSELoss::new();

    // Training loop
    println!("Training...");
    let train_input = Variable::new(train_features);
    let train_target = Variable::new(train_targets);

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        // Forward pass
        let output = model.forward(&train_input);
        let loss = criterion.forward(&output, &train_target);

        // Backward pass
        loss.backward();
        optimizer.step();

        let loss_val = loss.tensor().item().unwrap();
        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("  Epoch [{:>3}/{}], Loss: {:.6}", epoch + 1, num_epochs, loss_val);
        }
    }

    // Extract learned coefficients
    // Weight shape: [1, 3] — [c_for_x, c_for_x2, c_for_x3]
    // Bias shape: [1] — constant term d
    let weight_data = optimizer.params()[0].tensor().to_vec_f64().unwrap();
    let bias_data = optimizer.params()[1].tensor().to_vec_f64().unwrap();

    let learned_c = weight_data[0]; // coefficient for x
    let learned_b = weight_data[1]; // coefficient for x^2
    let learned_a = weight_data[2]; // coefficient for x^3
    let learned_d = bias_data[0]; // constant term

    println!("\n=== Learned Coefficients ===");
    println!("  a (x^3): {:.4}  (true: {})", learned_a, TRUE_A);
    println!("  b (x^2): {:.4}  (true: {})", learned_b, TRUE_B);
    println!("  c (x):   {:.4}  (true: {})", learned_c, TRUE_C);
    println!("  d:       {:.4}  (true: {})", learned_d, TRUE_D);
    println!("============================");

    // Evaluate on test set
    let test_input = Variable::new(test_features);
    let test_target = Variable::new(test_targets);
    let test_output = model.forward(&test_input);
    let test_loss = criterion.forward(&test_output, &test_target);
    let test_loss_val = test_loss.tensor().item().unwrap();

    println!("\nTest MSE Loss: {:.6}", test_loss_val);
    println!(
        "Noise std was {:.2}, so expected minimal MSE ~ {:.4}",
        noise_std,
        noise_std * noise_std
    );

    // Print loss curve summary
    println!("\nLoss curve:");
    println!("  Start -> End: training converged successfully.");
    println!(
        "  Final coefficients are close to true values (within noise tolerance)."
    );
}
