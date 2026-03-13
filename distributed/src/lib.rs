//! Distributed Data Parallel model definitions.
//!
//! Simple feedforward classifier:
//! - Linear(128, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)

use std::collections::HashMap;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_nn::{Linear, Module};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct SimpleModel {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

impl SimpleModel {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(128, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 10),
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x).relu().unwrap();
        let h = self.fc2.forward(&h).relu().unwrap();
        self.fc3.forward(&h)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
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
        for (name, param) in self.fc3.named_parameters() {
            sd.insert(format!("fc3.{name}"), param.tensor().clone());
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
            fc3: Linear::from_tensors(
                sd["fc3.weight"].clone(),
                Some(sd["fc3.bias"].clone()),
            ),
        }
    }
}
