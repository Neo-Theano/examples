//! Graph Attention Network (GAT) Example
//!
//! Trains a GAT with multi-head attention for node classification on a synthetic graph.
//! Saves the model to `gat_model.safetensors`.

use theano_autograd::Variable;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use gat::{GAT, random_adjacency_with_self_loops, random_features, random_labels};

fn main() {
    println!("=== Graph Attention Network (GAT) — Node Classification ===\n");

    let num_nodes = 50;
    let num_features = 16;
    let hidden_dim = 8;
    let num_heads = 8;
    let num_classes = 4;
    let num_epochs = 30;
    let lr = 0.005;
    let edge_prob = 0.15;

    let adj = random_adjacency_with_self_loops(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj);
    let feat_var = Variable::new(features);
    let label_var = Variable::new(labels);

    let model = GAT::new(num_features, hidden_dim, num_classes, num_heads);
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), lr);

    let total_params: usize = model.parameters().iter().map(|p| p.tensor().numel()).sum();
    println!(
        "Graph: {} nodes, {} features, {} classes, {} attention heads",
        num_nodes, num_features, num_classes, num_heads
    );
    println!(
        "Model: GATLayer({}, {}, heads={}) -> ReLU -> GATLayer({}, {}, heads=1)",
        num_features, hidden_dim, num_heads, num_heads * hidden_dim, num_classes
    );
    println!("Total parameters: {}\n", total_params);

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        let logits = model.forward(&adj_var, &feat_var);
        let loss = criterion.forward(&logits, &label_var);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "Epoch [{:>3}/{}]  Loss: {:.4}",
                epoch + 1,
                num_epochs,
                loss_val
            );
        }
    }

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("gat_model.safetensors", bytes).unwrap();

    println!("\nTraining complete. Model saved to gat_model.safetensors");
}
