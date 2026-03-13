//! FX Graph Mode Example
//!
//! Demonstrates graph capture, IR construction, optimisation passes (dead code
//! elimination), and tracing using theano-jit.

use theano_jit::{Graph, Op, Value, trace};
use theano_jit::passes::dead_code_elimination;

fn main() {
    println!("=== FX Graph Mode — theano-jit ===\n");

    // -----------------------------------------------------------------------
    // 1. Build a computation graph manually
    // -----------------------------------------------------------------------
    println!("--- Manual Graph Construction ---\n");

    let mut graph = Graph::new();

    // Inputs
    let x = graph.add_node(
        Op::Constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        vec![2, 2],
    );
    let w = graph.add_node(
        Op::Constant(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]),
        vec![2, 2],
    );

    // Computation: y = relu(x @ w)
    let mm = graph.add_node(Op::MatMul(x, w), vec![2, 2]);
    let y = graph.add_node(Op::Relu(mm), vec![2, 2]);

    // Some dead code (not reachable from output)
    let dead_const = graph.add_node(
        Op::Constant(vec![99.0, 99.0], vec![2]),
        vec![2],
    );
    let dead_neg = graph.add_node(Op::Neg(dead_const), vec![2]);
    let _dead_exp = graph.add_node(Op::Exp(dead_neg), vec![2]);

    // More dead code branching from a live node
    let _dead_branch = graph.add_node(Op::Sigmoid(mm), vec![2, 2]);

    // Final output: loss = mean(y)
    let loss = graph.add_node(Op::Mean(y), vec![]);

    graph.set_outputs(vec![loss]);

    println!("Graph BEFORE optimisation ({} nodes):", graph.len());
    println!("{}", graph.dump());

    // -----------------------------------------------------------------------
    // 2. Dead Code Elimination
    // -----------------------------------------------------------------------
    println!("--- Dead Code Elimination ---\n");

    let optimised = dead_code_elimination(&graph);

    println!("Graph AFTER optimisation ({} nodes):", optimised.len());
    println!("{}", optimised.dump());

    println!(
        "Removed {} dead nodes.\n",
        graph.len() - optimised.len()
    );

    // -----------------------------------------------------------------------
    // 3. More complex graph: neural network forward pass
    // -----------------------------------------------------------------------
    println!("--- Neural Network Graph ---\n");

    let mut nn_graph = Graph::new();

    // Input and weights
    let input = nn_graph.add_node(
        Op::Constant(vec![0.0; 8], vec![2, 4]),
        vec![2, 4],
    );
    let w1 = nn_graph.add_node(
        Op::Constant(vec![0.0; 12], vec![4, 3]),
        vec![4, 3],
    );
    let b1 = nn_graph.add_node(
        Op::Constant(vec![0.0; 3], vec![3]),
        vec![3],
    );
    let w2 = nn_graph.add_node(
        Op::Constant(vec![0.0; 6], vec![3, 2]),
        vec![3, 2],
    );

    // Layer 1: h = relu(input @ w1 + b1)
    let mm1 = nn_graph.add_node(Op::MatMul(input, w1), vec![2, 3]);
    let add1 = nn_graph.add_node(Op::Add(mm1, b1), vec![2, 3]);
    let h = nn_graph.add_node(Op::Relu(add1), vec![2, 3]);

    // Layer 2: output = h @ w2
    let output = nn_graph.add_node(Op::MatMul(h, w2), vec![2, 2]);

    // Loss = mean(output)
    let nn_loss = nn_graph.add_node(Op::Mean(output), vec![]);

    nn_graph.set_outputs(vec![nn_loss]);

    println!("Neural network graph ({} nodes):", nn_graph.len());
    println!("{}", nn_graph.dump());

    // -----------------------------------------------------------------------
    // 4. Graph tracing
    // -----------------------------------------------------------------------
    println!("--- Graph Tracing ---\n");

    let (traced_graph, traced_inputs) = trace(
        &[vec![4, 3], vec![3, 2]],
        |tensors| {
            let a = &tensors[0];
            let b = &tensors[1];
            a.matmul(b).unwrap()
        },
    );

    println!("Traced graph ({} nodes):", traced_graph.len());
    println!("{}", traced_graph.dump());
    println!("Traced inputs: {:?}", traced_inputs);

    // -----------------------------------------------------------------------
    // 5. Demonstrate various IR operations
    // -----------------------------------------------------------------------
    println!("\n--- IR Operations Catalogue ---\n");

    let mut demo = Graph::new();
    let a = demo.add_node(Op::Constant(vec![1.0, 2.0], vec![2]), vec![2]);
    let b = demo.add_node(Op::Constant(vec![3.0, 4.0], vec![2]), vec![2]);

    let add_v = demo.add_node(Op::Add(a, b), vec![2]);
    let sub_v = demo.add_node(Op::Sub(a, b), vec![2]);
    let mul_v = demo.add_node(Op::Mul(a, b), vec![2]);
    let div_v = demo.add_node(Op::Div(a, b), vec![2]);
    let neg_v = demo.add_node(Op::Neg(a), vec![2]);
    let exp_v = demo.add_node(Op::Exp(a), vec![2]);
    let log_v = demo.add_node(Op::Log(a), vec![2]);
    let sqrt_v = demo.add_node(Op::Sqrt(a), vec![2]);
    let tanh_v = demo.add_node(Op::Tanh(a), vec![2]);
    let sigmoid_v = demo.add_node(Op::Sigmoid(a), vec![2]);
    let relu_v = demo.add_node(Op::Relu(a), vec![2]);
    let sum_v = demo.add_node(Op::Sum(a), vec![]);
    let mean_v = demo.add_node(Op::Mean(a), vec![]);
    let reshape_v = demo.add_node(Op::Reshape(a, vec![1, 2]), vec![1, 2]);
    let mat = demo.add_node(Op::Constant(vec![0.0; 6], vec![2, 3]), vec![2, 3]);
    let transpose_v = demo.add_node(Op::Transpose(mat, 0, 1), vec![3, 2]);

    let ops: Vec<(&str, Value)> = vec![
        ("add", add_v),
        ("sub", sub_v),
        ("mul", mul_v),
        ("div", div_v),
        ("neg", neg_v),
        ("exp", exp_v),
        ("log", log_v),
        ("sqrt", sqrt_v),
        ("tanh", tanh_v),
        ("sigmoid", sigmoid_v),
        ("relu", relu_v),
        ("sum", sum_v),
        ("mean", mean_v),
        ("reshape", reshape_v),
        ("transpose", transpose_v),
    ];

    for (name, val) in &ops {
        let node = demo.get_node(*val).unwrap();
        println!("  {:>10}: {} = {}  (shape={:?})", name, node.id, node.op, node.shape);
    }

    println!("\nFX graph mode demonstration complete.");
}
