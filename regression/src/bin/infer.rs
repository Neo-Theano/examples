//! Polynomial Regression inference example.
//!
//! Loads a trained model and predicts y for sample x values.
//! Prints the learned polynomial coefficients.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::Module;
use theano_serialize::load_state_dict;

use regression::{PolynomialRegression, TRUE_A, TRUE_B, TRUE_C, TRUE_D};

fn main() {
    println!("=== Polynomial Regression Inference ===");
    println!();

    // Load trained model
    let bytes = std::fs::read("regression_model.safetensors")
        .expect("Model file not found. Run the training first: cargo run --bin regression");
    let sd = load_state_dict(&bytes).unwrap();
    let model = PolynomialRegression::from_state_dict(&sd);
    println!("Model loaded from regression_model.safetensors");

    // Extract learned coefficients
    let weight_data = model.linear.named_parameters();
    let w = weight_data.iter().find(|(n, _)| n == "weight").unwrap().1.tensor().to_vec_f64().unwrap();
    let b = weight_data.iter().find(|(n, _)| n == "bias").unwrap().1.tensor().to_vec_f64().unwrap();

    let learned_c = w[0]; // coefficient for x
    let learned_b = w[1]; // coefficient for x^2
    let learned_a = w[2]; // coefficient for x^3
    let learned_d = b[0]; // constant term

    println!("\n--- Learned Coefficients ---");
    println!("  a (x^3): {:.4}  (true: {})", learned_a, TRUE_A);
    println!("  b (x^2): {:.4}  (true: {})", learned_b, TRUE_B);
    println!("  c (x):   {:.4}  (true: {})", learned_c, TRUE_C);
    println!("  d:       {:.4}  (true: {})", learned_d, TRUE_D);

    // Predict y for sample x values
    println!("\n--- Predictions ---");
    let sample_xs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    for &x in &sample_xs {
        let x2 = x * x;
        let x3 = x2 * x;
        let features = Tensor::from_slice(&[x, x2, x3], &[1, 3]);
        let input = Variable::new(features);
        let output = model.forward(&input);
        let predicted = output.tensor().item().unwrap();
        let true_y = TRUE_A * x3 + TRUE_B * x2 + TRUE_C * x + TRUE_D;
        println!(
            "  x = {:5.1}  ->  predicted: {:7.4},  true: {:7.4}",
            x, predicted, true_y
        );
    }

    println!("\nInference complete.");
}
