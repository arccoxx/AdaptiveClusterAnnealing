# -*- coding: utf-8 -*-
"""
The Definitive Ising Machine: The Adaptive Cluster Annealing (ACA) Solver

This script benchmarks the ACA solver against a standard baseline, Simulated Annealing (SA),
for the Maximum Cut (MAX-CUT) problem. This provides a clear and robust comparison
of the two heuristic methods.
"""

import numpy as np
import time
import math

try:
    import cupy as cp
    print("✅ CuPy (GPU) backend found.")
except ImportError:
    print("⚠️ CuPy not found. Run 'pip install cupy-cudaXXX' (e.g., cupy-cuda12x).")
    exit()

# For graph generation and visualization
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    print("✅ NetworkX and Matplotlib found for visualization.")
except ImportError:
    print("⚠️ NetworkX/Matplotlib not found. Run 'pip install networkx matplotlib'.")
    exit()

# =============================================================================
# 1. The Validated CUDA Kernels for the ACA Solver
# =============================================================================
ACA_CUDA_MODULE = cp.RawModule(code=r'''
/* ... CUDA kernels are unchanged from the previous version ... */
extern "C" {
    __device__ unsigned int xorshift32(unsigned int& state) { unsigned int x = state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; state = x; return x; }
    __global__ void evolution_kernel(double* spins, const double* J, const double* h, const double* temperatures, double* energies, const int num_spins, unsigned int* rand_states, const double* rand_uniform) {
        const int r = blockIdx.x; unsigned int thread_rand_state = rand_states[r];
        for (int sweep = 0; sweep < num_spins; ++sweep) {
            int i = xorshift32(thread_rand_state) % num_spins; double energy_change = 0.0;
            for (int j = 0; j < num_spins; ++j) { energy_change += 2.0 * spins[r * num_spins + i] * J[i * num_spins + j] * spins[r * num_spins + j]; }
            energy_change += 2.0 * spins[r * num_spins + i] * h[i];
            if (energy_change < 0 || rand_uniform[r * num_spins + sweep] < exp(-energy_change / temperatures[r])) { spins[r * num_spins + i] *= -1.0; energies[r] += energy_change; }
        }
        rand_states[r] = thread_rand_state;
    }
    __global__ void correlation_kernel(const double* spins, const int num_spins, const int num_replicas, double* output_correlations) {
        const int i = blockIdx.x; const int j = threadIdx.x; if (i >= num_spins || j >= num_spins) return;
        double corr_sum = 0.0;
        for (int r = 0; r < num_replicas; ++r) { corr_sum += spins[r * num_spins + i] * spins[r * num_spins + j]; }
        output_correlations[i * num_spins + j] = corr_sum / num_replicas;
    }
    __global__ void adaptive_entangled_kernel(double* spins, const double* J, const double* h, const double* temperatures, double* energies, const double* correlation_matrix, const double* thresholds, int* acceptance_counts, const int num_spins, unsigned int* rand_states, const double* rand_uniform) {
        const int r = blockIdx.x; unsigned int thread_rand_state = rand_states[r]; double replica_threshold = thresholds[r];
        for (int sweep = 0; sweep < 5; ++sweep) { 
            int i = xorshift32(thread_rand_state) % num_spins;
            int cluster[128]; int cluster_size = 0; cluster[cluster_size++] = i;
            for (int j = 0; j < num_spins; ++j) { if (i != j && abs(correlation_matrix[i * num_spins + j]) > replica_threshold) { if (cluster_size < 128) cluster[cluster_size++] = j; } }
            if (cluster_size <= 1) continue;
            double block_delta_E = 0.0;
            for (int c_idx = 0; c_idx < cluster_size; ++c_idx) {
                int s_idx = cluster[c_idx];
                block_delta_E -= 2.0 * spins[r * num_spins + s_idx] * h[s_idx];
                for (int j = 0; j < num_spins; ++j) {
                    bool in_cluster = false;
                    for (int k_idx = 0; k_idx < cluster_size; ++k_idx) { if (cluster[k_idx] == j) { in_cluster = true; break; } }
                    if (!in_cluster) { block_delta_E -= 2.0 * spins[r * num_spins + s_idx] * J[s_idx * num_spins + j] * spins[r * num_spins + j]; }
                }
            }
            if (block_delta_E < 0 || rand_uniform[r * num_spins + sweep] < exp(-block_delta_E / temperatures[r])) {
                for (int c_idx = 0; c_idx < cluster_size; ++c_idx) { spins[r * num_spins + cluster[c_idx]] *= -1.0; }
                energies[r] += block_delta_E; atomicAdd(&acceptance_counts[r], 1);
            }
        }
        rand_states[r] = thread_rand_state;
    }
}
''')

# =============================================================================
# 2. The Solver Classes
# =============================================================================
class AdaptiveClusterAnnealer:
    def __init__(self, J, h, num_replicas=24, T_min=0.01, T_max=10.0):
        self.num_spins, self.num_replicas = len(h), num_replicas; self.J_gpu, self.h_gpu = cp.asarray(J, dtype=cp.float64), cp.asarray(h, dtype=cp.float64)
        self.temps_gpu = cp.asarray(np.geomspace(T_min, T_max, num_replicas, dtype=np.float64))
        self.kernels = {name: ACA_CUDA_MODULE.get_function(name) for name in ["evolution_kernel", "correlation_kernel", "adaptive_entangled_kernel"]}
    def solve(self, cycles=100, steps_per_cycle=50):
        print("\n--- Running Solver 1: Adaptive Cluster Annealing (ACA) ---"); spins = cp.random.choice(cp.array([-1, 1], dtype=cp.float64), size=(self.num_replicas, self.num_spins))
        energies = cp.array([-0.5*cp.sum(self.J_gpu*cp.outer(s,s)) - cp.sum(self.h_gpu*s) for s in spins]); C = cp.zeros((self.num_spins, self.num_spins), dtype=cp.float64)
        rand_states = cp.random.randint(1, 2**31, size=self.num_replicas, dtype=cp.uint32); thresholds = cp.full((self.num_replicas,), 0.6, dtype=cp.float64)
        grid_evo, block_evo, grid_corr, block_corr = (self.num_replicas,), (1,), (self.num_spins,), (self.num_spins,)
        for i in range(cycles):
            for _ in range(steps_per_cycle): self.kernels["evolution_kernel"](grid_evo, block_evo, (spins, self.J_gpu, self.h_gpu, self.temps_gpu, energies, self.num_spins, rand_states, cp.random.rand(self.num_replicas, self.num_spins)))
            temp_C = cp.zeros_like(C); self.kernels["correlation_kernel"](grid_corr, block_corr, (spins, self.num_spins, self.num_replicas, temp_C)); C = 0.95 * C + 0.05 * temp_C
            accept_counts = cp.zeros(self.num_replicas, dtype=cp.int32)
            self.kernels["adaptive_entangled_kernel"](grid_evo, block_evo, (spins, self.J_gpu, self.h_gpu, self.temps_gpu, energies, C, thresholds, accept_counts, self.num_spins, rand_states, cp.random.rand(self.num_replicas, self.num_spins)))
            accept_rates = accept_counts.get() / 5.0; thresholds_cpu = thresholds.get(); thresholds_cpu[accept_rates < 0.2] += 0.01; thresholds_cpu[accept_rates > 0.5] -= 0.01; thresholds = cp.asarray(np.clip(thresholds_cpu, 0.2, 0.9))
            for r_start in [0, 1]:
                for r in range(r_start, self.num_replicas - 1, 2):
                    e1, e2 = energies[r], energies[r+1]; t1, t2 = self.temps_gpu[r], self.temps_gpu[r+1]
                    if cp.random.rand() < cp.exp((e1 - e2) * (1/t1 - 1/t2)): spins[r], spins[r+1] = spins[r+1].copy(), spins[r].copy(); energies[r], energies[r+1] = e2, e1
            if (i + 1) % 10 == 0: print(f"  ACA Cycle {i+1}/{cycles}, Best Energy: {energies[0].get():.4f}")
        return spins[0].get()

def solve_max_cut_simulated_annealing(G, J, h, steps=100000):
    """
    Solves the MAX-CUT problem using a standard Simulated Annealing algorithm.
    """
    print("\n--- Running Solver 2: Simulated Annealing (Standard Baseline) ---")
    num_nodes = G.number_of_nodes()
    
    # Initialize state
    spins = np.random.choice([-1, 1], size=num_nodes)
    current_energy = -0.5 * spins.T @ J @ spins
    best_spins = spins.copy()
    best_energy = current_energy
    
    # Annealing schedule
    T_initial = 5.0
    T_final = 0.01
    cooling_rate = (T_final / T_initial)**(1.0 / steps)
    T = T_initial
    
    for i in range(steps):
        # Propose a local move (flip one random spin)
        flip_idx = np.random.randint(0, num_nodes)
        
        # Calculate energy change efficiently
        delta_E = 0
        for j in range(num_nodes):
            delta_E += 2 * spins[flip_idx] * J[flip_idx, j] * spins[j]
        
        # Metropolis acceptance criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            spins[flip_idx] *= -1
            current_energy += delta_E
            if current_energy < best_energy:
                best_energy = current_energy
                best_spins = spins.copy()
        
        # Cool down
        T *= cooling_rate
        if (i + 1) % 20000 == 0:
            print(f"  SA Step {i+1}/{steps}, Best Energy: {best_energy:.4f}")
            
    return best_spins

# =============================================================================
# 3. Problem Formulation and Evaluation
# =============================================================================
def create_max_cut_ising_model(G):
    num_nodes = G.number_of_nodes()
    J = np.zeros((num_nodes, num_nodes)); h = np.zeros(num_nodes)
    for i, j in G.edges(): J[i, j] = -1.0; J[j, i] = -1.0
    return J, h

def evaluate_cut(spins, G):
    return sum(1 for i, j in G.edges() if spins[i] != spins[j])

# =============================================================================
# 4. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    num_nodes = 50
    print(f"--- Generating a {num_nodes}-node MAX-CUT problem ---")
    G = nx.erdos_renyi_graph(num_nodes, p=0.3, seed=42)
    J, h = create_max_cut_ising_model(G)
    
    # --- Run ACA Solver ---
    aca_solver = AdaptiveClusterAnnealer(J, h)
    start_time = time.time()
    aca_spins = aca_solver.solve(cycles=200, steps_per_cycle=50)
    aca_time = time.time() - start_time
    aca_cut = evaluate_cut(aca_spins, G)
    
    # --- Run Simulated Annealing Solver ---
    start_time = time.time()
    sa_spins = solve_max_cut_simulated_annealing(G, J, h)
    sa_time = time.time() - start_time
    sa_cut = evaluate_cut(sa_spins, G) if sa_spins is not None else -1

    # --- Comparison Report ---
    print("\n" + "="*60); print("--- Comparison Report: ACA vs. Simulated Annealing ---"); print("="*60)
    print(f"{'Metric':<25} | {'Adaptive Cluster Annealer':<25} | {'Simulated Annealing (SA)'}")
    print("-"*60)
    print(f"{'Cut Size (Edges)':<25} | {aca_cut:<25} | {sa_cut}")
    print(f"{'% of Total Edges Cut':<25} | {100 * aca_cut / len(G.edges()):<25.2f}% | {100 * sa_cut / len(G.edges()):.2f}%")
    print(f"{'Solver Time (seconds)':<25} | {aca_time:<25.4f} | {sa_time:.4f}")
    print("="*60)
    
    # --- Visualize the Best Result ---
    best_spins = aca_spins if aca_cut > sa_cut else sa_spins
    best_method = "Adaptive Cluster Annealing" if aca_cut > sa_cut else "Simulated Annealing"
    best_cut_size = max(aca_cut, sa_cut)
    
    node_colors = ['skyblue' if spin == 1 else 'tomato' for spin in best_spins]
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_color='black', edge_color='gray')
    plt.title(f"Best MAX-CUT Solution via {best_method} ({best_cut_size} edges cut)", fontsize=16)
    plt.show()
