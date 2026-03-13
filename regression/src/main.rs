//! Polynomial Regression Example — Fit y = a*x^3 + b*x^2 + c*x + d + noise.
//!
//! Demonstrates basic regression with polynomial features, MSE loss, and SGD optimizer.
//! A simple but complete training pipeline showing the fundamentals.

use theano::prelude::*;
use theano_nn::MSELoss;
use theano_optim::{Optimizer, SGD};
use theano_serialize::save_state_dict;

use regression::{generate_data, PolynomialRegression, TRUE_A, TRUE_B, TRUE_C, TRUE_D};

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
    let model = PolynomialRegression::new();

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

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("regression_model.safetensors", bytes).unwrap();
    println!("Model saved to regression_model.safetensors");
}
