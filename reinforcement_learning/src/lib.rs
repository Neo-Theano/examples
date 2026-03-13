//! Reinforcement Learning — REINFORCE (Policy Gradient) model definitions.
//!
//! Policy network: Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_actions) -> Softmax

use std::collections::HashMap;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// Hyperparameters
pub const STATE_DIM: usize = 4;
pub const NUM_ACTIONS: usize = 5;
pub const HIDDEN_DIM: usize = 128;
pub const GAMMA: f64 = 0.99;
pub const NUM_EPISODES: usize = 200;
pub const EPISODE_BATCH_SIZE: usize = 10;
pub const MAX_STEPS: usize = 20;
pub const LEARNING_RATE: f64 = 0.01;

/// Policy network for REINFORCE.
pub struct PolicyNetwork {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl PolicyNetwork {
    pub fn new(state_dim: usize, hidden_dim: usize, num_actions: usize) -> Self {
        Self {
            fc1: Linear::new(state_dim, hidden_dim),
            fc2: Linear::new(hidden_dim, num_actions),
        }
    }

    /// Forward pass: state [1, state_dim] -> action probabilities [1, num_actions]
    pub fn forward(&self, state: &Variable) -> Variable {
        let h = self.fc1.forward(state).relu().unwrap();
        let logits = self.fc2.forward(&h);
        logits.softmax(-1).unwrap()
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, param) in self.fc1.named_parameters() {
            sd.insert(format!("fc1.{name}"), param.tensor().clone());
        }
        for (name, param) in self.fc2.named_parameters() {
            sd.insert(format!("fc2.{name}"), param.tensor().clone());
        }
        sd
    }

    pub fn from_state_dict(sd: &HashMap<String, Tensor>) -> Self {
        Self {
            fc1: Linear::from_tensors(
                sd["fc1.weight"].clone(),
                Some(sd["fc1.bias"].clone()),
            ),
            fc2: Linear::from_tensors(
                sd["fc2.weight"].clone(),
                Some(sd["fc2.bias"].clone()),
            ),
        }
    }
}

/// Sample an action from the probability distribution.
pub fn sample_action(probs: &[f64]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rand::Rng::gen(&mut rng);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Compute discounted returns from a sequence of rewards.
pub fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    let mut returns = vec![0.0; n];
    let mut running = 0.0;
    for t in (0..n).rev() {
        running = rewards[t] + gamma * running;
        returns[t] = running;
    }

    // Normalize returns for stability
    if n > 1 {
        let mean: f64 = returns.iter().sum::<f64>() / n as f64;
        let variance: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        let std = (variance + 1e-8).sqrt();
        for r in returns.iter_mut() {
            *r = (*r - mean) / std;
        }
    }

    returns
}
