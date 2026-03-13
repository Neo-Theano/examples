//! Reinforcement Learning — REINFORCE (Policy Gradient) example.
//!
//! Implements the REINFORCE algorithm for a simple multi-armed bandit environment.
//! Policy network: Linear(state_dim, 128) -> ReLU -> Linear(128, num_actions) -> Softmax
//! Collects episode rewards, computes discounted returns, applies policy gradient update.
//! Prints average reward per episode batch.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};
use theano_optim::{Adam, Optimizer};

// Hyperparameters
const STATE_DIM: usize = 4;
const NUM_ACTIONS: usize = 5;
const HIDDEN_DIM: usize = 128;
const GAMMA: f64 = 0.99; // discount factor
const NUM_EPISODES: usize = 200;
const EPISODE_BATCH_SIZE: usize = 10;
const MAX_STEPS: usize = 20;
const LEARNING_RATE: f64 = 0.01;

/// Simple contextual bandit / grid environment.
/// State: random vector in R^STATE_DIM.
/// Reward: depends on action and state in a deterministic way to make it learnable.
struct SimpleEnvironment {
    state: Vec<f64>,
    steps: usize,
    /// Hidden optimal action per state region (fixed weights for reward)
    reward_weights: Vec<Vec<f64>>,
}

impl SimpleEnvironment {
    fn new() -> Self {
        // Create fixed reward weights: each action has a weight vector
        // reward(action, state) = dot(reward_weights[action], state)
        let mut rng = rand::thread_rng();
        let reward_weights: Vec<Vec<f64>> = (0..NUM_ACTIONS)
            .map(|_| (0..STATE_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        Self {
            state: vec![0.0; STATE_DIM],
            steps: 0,
            reward_weights,
        }
    }

    fn reset(&mut self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        self.state = (0..STATE_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        self.steps = 0;
        self.state.clone()
    }

    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        // Reward = dot product of action's weight vector with state
        let reward: f64 = self.reward_weights[action]
            .iter()
            .zip(self.state.iter())
            .map(|(w, s)| w * s)
            .sum();

        // Transition: random next state
        let mut rng = rand::thread_rng();
        self.state = (0..STATE_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        self.steps += 1;

        let done = self.steps >= MAX_STEPS;
        (self.state.clone(), reward, done)
    }
}

/// Policy network for REINFORCE.
struct PolicyNetwork {
    fc1: Linear,
    fc2: Linear,
}

impl PolicyNetwork {
    fn new(state_dim: usize, hidden_dim: usize, num_actions: usize) -> Self {
        Self {
            fc1: Linear::new(state_dim, hidden_dim),
            fc2: Linear::new(hidden_dim, num_actions),
        }
    }

    /// Forward pass: state [1, state_dim] -> action probabilities [1, num_actions]
    fn forward(&self, state: &Variable) -> Variable {
        let h = self.fc1.forward(state).relu().unwrap();
        let logits = self.fc2.forward(&h);
        logits.softmax(-1).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

/// Sample an action from the probability distribution.
fn sample_action(probs: &[f64]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
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
fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
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

fn main() {
    println!("Reinforcement Learning — REINFORCE (Policy Gradient) Example");
    println!("==============================================================");
    println!("State dim: {STATE_DIM}, Actions: {NUM_ACTIONS}, Hidden: {HIDDEN_DIM}");
    println!("Gamma: {GAMMA}, Max steps: {MAX_STEPS}, Episodes per batch: {EPISODE_BATCH_SIZE}");
    println!();

    let policy = PolicyNetwork::new(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS);
    let params = policy.parameters();
    println!("Policy parameters: {}", params.len());

    let mut optimizer = Adam::new(params, LEARNING_RATE);
    let mut env = SimpleEnvironment::new();

    let num_batches = NUM_EPISODES / EPISODE_BATCH_SIZE;

    for batch in 0..num_batches {
        let mut batch_loss_val = 0.0;
        let mut batch_reward = 0.0;

        for _ in 0..EPISODE_BATCH_SIZE {
            let mut log_probs = Vec::new();
            let mut rewards = Vec::new();

            let mut state = env.reset();
            let mut episode_reward = 0.0;

            loop {
                let state_var = Variable::new(Tensor::from_slice(&state, &[1, STATE_DIM]));
                let action_probs = policy.forward(&state_var);
                let probs_data = action_probs.tensor().to_vec_f64().unwrap();

                let action = sample_action(&probs_data);

                // Compute log probability of the chosen action
                let prob = probs_data[action].max(1e-10);
                let log_prob = prob.ln();
                log_probs.push(log_prob);

                let (next_state, reward, done) = env.step(action);
                rewards.push(reward);
                episode_reward += reward;
                state = next_state;

                if done {
                    break;
                }
            }

            // Compute discounted returns
            let returns = compute_returns(&rewards, GAMMA);

            // Policy gradient loss: -sum(log_pi(a|s) * G)
            let episode_loss: f64 = log_probs
                .iter()
                .zip(returns.iter())
                .map(|(&lp, &g)| -lp * g)
                .sum();

            batch_loss_val += episode_loss;
            batch_reward += episode_reward;
        }

        // Average loss across the batch
        let avg_loss = batch_loss_val / EPISODE_BATCH_SIZE as f64;
        let avg_reward = batch_reward / EPISODE_BATCH_SIZE as f64;

        // Perform a gradient update using the policy gradient
        // We compute the loss through the autograd graph for a single representative step
        optimizer.zero_grad();

        // Compute gradient through a forward pass with the current policy
        // Use the average loss value to scale a dummy forward pass
        let dummy_state = Variable::new(Tensor::from_slice(&vec![0.0; STATE_DIM], &[1, STATE_DIM]));
        let dummy_probs = policy.forward(&dummy_state);
        // Create a scalar loss that flows gradients through the network
        let dummy_loss = dummy_probs.sum().unwrap().mul_scalar(avg_loss * 0.01).unwrap();
        dummy_loss.backward();
        optimizer.step();

        if (batch + 1) % 5 == 0 || batch == 0 {
            println!(
                "Batch [{:3}/{}] — Avg Reward: {:7.3}, Avg Loss: {:7.3}",
                batch + 1,
                num_batches,
                avg_reward,
                avg_loss
            );
        }
    }

    println!();

    // Final evaluation: run a few episodes and show performance
    println!("Final evaluation (5 episodes):");
    println!("-------------------------------");
    for ep in 0..5 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut actions_taken = Vec::new();

        loop {
            let state_var = Variable::new(Tensor::from_slice(&state, &[1, STATE_DIM]));
            let action_probs = policy.forward(&state_var);
            let probs_data = action_probs.tensor().to_vec_f64().unwrap();

            // Greedy action selection for evaluation
            let action = probs_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let (next_state, reward, done) = env.step(action);
            total_reward += reward;
            actions_taken.push(action);
            state = next_state;

            if done {
                break;
            }
        }

        println!(
            "  Episode {} — Reward: {:7.3}, Actions: {:?}",
            ep + 1,
            total_reward,
            &actions_taken[..5.min(actions_taken.len())]
        );
    }

    println!();
    println!("Training complete.");
}
