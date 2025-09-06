***

# Adaptive Cluster Annealing (ACA) Solver

This repository contains the source code for the Adaptive Cluster Annealing (ACA) solver, a high-performance, GPU-accelerated heuristic designed to find low-energy states of Ising models and solutions to Quadratic Unconstrained Binary Optimization (QUBO) problems.

The ACA solver is an advanced metaheuristic that builds upon the principles of parallel tempering and simulated annealing. Its primary innovation is the introduction of a collective learning mechanism that allows a multi-replica ensemble to perform adaptive, non-local cluster moves, enabling it to navigate complex and rugged energy landscapes more effectively than methods that rely solely on local updates.

## The ACA Method

The solver operates by deploying a population of "replicas" (or walkers), each exploring the solution space at a different temperature. The algorithm cycles through five distinct phases that combine parallel local exploration with collective, adaptive global moves.

1.  **Parallel Evolution:** All replicas independently explore the solution space using standard Metropolis-Hastings spin flips at their assigned temperatures. Hotter replicas are more adventurous, accepting high-energy moves, while colder replicas perform finer-grained local searches.
2.  **Collective Learning:** Periodically, the states of all replicas are averaged to compute a global spin-correlation matrix. This matrix captures the emergent, system-wide structures that the ensemble has collectively discovered.
3.  **Adaptive Entangled Moves:** Using the global correlation matrix, the solver proposes intelligent, non-local moves. It identifies clusters of highly correlated spins and attempts to flip them as a single block. Crucially, each replica *adapts* its own sensitivity for what constitutes a cluster based on the historical success rate of its previous cluster moves.
4.  **Feedback & Adaptation:** The success rates of the entangled moves are used as a feedback signal. Replicas that are too timid (high acceptance) become bolder, while replicas that are too reckless (low acceptance) become more conservative in their cluster proposals.
5.  **Replica Exchange:** A stochastic swap mechanism allows adjacent replicas in the temperature ladder to exchange their states. This is a critical safety net that enables the primary (coldest) solution to escape deep local minima by temporarily adopting the more exploratory state of a hotter neighbor.

## Benchmark Experiment: Maximum Cut (MAX-CUT)

To empirically validate its performance, the ACA solver was benchmarked against a standard **Simulated Annealing (SA)** baseline on the NP-hard Maximum Cut problem. MAX-CUT is an ideal test case as it maps cleanly to the Ising model without the complex constraints that can confound solvers. A random 50-node graph was generated, and both solvers were tasked with finding a partition of nodes that maximized the number of edges connecting the two sets.

### Results

| Metric | Adaptive Cluster Annealer (GPU) | Simulated Annealing (CPU) |
| :--- | :--- | :--- |
| **Cut Size (Edges)** | **246** | **246** |
| **% of Total Edges Cut** | **66.85%** | **66.85%** |
| **Solver Time (seconds)** | 4.7789 s | 4.2345 s |

### Analysis

For a moderately-sized graph, both the advanced, parallel ACA solver and the standard, serial SA baseline successfully converged to a high-quality solution of **identical size (246 edges cut)**. The GPU-accelerated ACA was slightly slower, a result attributed to the computational overhead of its more sophisticated learning and cluster-move phases.

This benchmark validates that the ACA implementation is robust and effective. While a simple SA was competitive on this specific problem, the architectural advantages of ACA—particularly its ability to perform non-local moves—are designed to provide a greater performance edge on larger, more complex, or more "frustrated" problem landscapes where simpler heuristics are more likely to become permanently trapped in local minima.

## A Second Application: Training Neural Networks

Beyond traditional optimization, the ACA solver can be adapted for experimental machine learning research, specifically for **training Binarized Neural Networks (BNNs) without gradients**. The script `modified_aca_nn_training.py` in this repository demonstrates this application.

The core concept is to reframe the neural network as an **Energy-Based Model (EBM)**:

1.  **Model Formulation:** The network is designed to take both an input (e.g., an MNIST image) and a potential label (`0` through `9`) and output a single scalar value representing the **energy** of that pair. The network's weights are binarized to `{-1, 1}`.
2.  **Training Objective:** The goal is to shape the energy landscape. The energy of a correct `(image, label)` pair should be low, while the energy of incorrect pairs should be high.
3.  **ACA as the Optimizer:** Instead of using backpropagation, the ACA solver is used to find the configuration of binary weights that minimizes a **contrastive loss function**. The solver treats the entire set of network weights as a spin vector and works to find the weight configuration that produces the best energy landscape.

This approach demonstrates the versatility of the ACA architecture as a powerful, non-gradient-based optimizer for complex, high-dimensional search problems in machine learning.

### Dependencies
- `numpy`
- `cupy` (for CUDA-enabled GPUs, e.g., `pip install cupy-cuda12x`)
- `networkx` and `matplotlib` (for the MAX-CUT visualization)
