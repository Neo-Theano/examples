//! Graph Convolutional Network (GCN) Example
//!
//! Trains a 2-layer GCN for node classification on a synthetic graph.
//! Saves the model to `gcn_model.safetensors`.

use theano_autograd::Variable;
use theano_nn::CrossEntropyLoss;
use theano_optim::{Adam, Optimizer};
use theano_serialize::save_state_dict;

use gcn::{
    GCN, random_normalized_adjacency, random_features, random_labels, compute_accuracy,
};

fn main() {
    println!("=== Graph Convolutional Network (GCN) — Node Classification ===\n");

    let num_nodes = 100;
    let num_features = 16;
    let hidden_dim = 16;
    let num_classes = 5;
    let num_epochs = 50;
    let lr = 0.01;
    let edge_prob = 0.1;

    // Generate synthetic graph
    let adj = random_normalized_adjacency(num_nodes, edge_prob);
    let features = random_features(num_nodes, num_features);
    let labels = random_labels(num_nodes, num_classes);

    let adj_var = Variable::new(adj);
    let feat_var = Variable::new(features);
    let label_var = Variable::new(labels.clone());

    let model = GCN::new(num_features, hidden_dim, num_classes);
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters(), lr);

    println!(
        "Graph: {} nodes, {} features, {} classes",
        num_nodes, num_features, num_classes
    );
    println!(
        "Model: GCNLayer({}, {}) -> ReLU -> GCNLayer({}, {})\n",
        num_features, hidden_dim, hidden_dim, num_classes
    );

    for epoch in 0..num_epochs {
        optimizer.zero_grad();

        let logits = model.forward(&adj_var, &feat_var);
        let loss = criterion.forward(&logits, &label_var);

        let loss_val = loss.tensor().item().unwrap();
        loss.backward();
        optimizer.step();

        let accuracy = compute_accuracy(logits.tensor(), &labels);

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Epoch [{:>3}/{}]  Loss: {:.4}  Accuracy: {:.2}%",
                epoch + 1,
                num_epochs,
                loss_val,
                accuracy * 100.0
            );
        }
    }

    // Save the trained model
    let sd = model.state_dict();
    let bytes = save_state_dict(&sd);
    std::fs::write("gcn_model.safetensors", bytes).unwrap();

    println!("\nTraining complete. Model saved to gcn_model.safetensors");
}
