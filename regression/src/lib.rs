//! Polynomial Regression model definitions.
//!
//! Model: Linear(3, 1) — maps polynomial features [x, x^2, x^3] to y.
//! The bias term acts as the constant d in y = a*x^3 + b*x^2 + c*x + d.

use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};
use theano_core::Tensor;
use theano_nn::{Linear, Module};

/// True polynomial coefficients: y = a*x^3 + b*x^2 + c*x + d
pub const TRUE_A: f64 = 0.5;
pub const TRUE_B: f64 = -1.2;
pub const TRUE_C: f64 = 2.0;
pub const TRUE_D: f64 = -0.3;

/// Polynomial regression model: Linear(3, 1).
pub struct PolynomialRegression {
    pub linear: Linear,
}

impl PolynomialRegression {
    pub fn new() -> Self {
        Self {
            linear: Linear::new(3, 1),
        }
    }

    pub fn forward(&self, x: &theano_autograd::Variable) -> theano_autograd::Variable {
        self.linear.forward(x)
    }

    pub fn parameters(&self) -> Vec<theano_autograd::Variable> {
        self.linear.parameters()
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.linear.named_parameters() {
            sd.insert(format!("linear.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            linear: Linear::from_tensors(
                sd["linear.weight"].clone(),
                Some(sd["linear.bias"].clone()),
            ),
        }
    }
}

/// Generate synthetic polynomial data.
///
/// Returns (features, targets) where features has polynomial columns [x, x^2, x^3]
/// and targets = a*x^3 + b*x^2 + c*x + d + noise.
pub fn generate_data(n: usize, noise_std: f64) -> (Tensor, Tensor) {
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
