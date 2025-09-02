The Adaptive Cluster Annealing (ACA) Ising Model Solver
This repository contains the source code for the Adaptive Cluster Annealing (ACA) solver, a high-performance, experimental Ising machine developed as part of a research journey into state-of-the-art heuristic optimization. The ACA architecture has been rigorously tested and proven to be the most powerful and robust solver of this research series, dramatically outperforming standard methods like Simulated Annealing and simpler replica-based approaches.

The solver is written in Python and uses a custom CUDA C++ kernel for massive parallelization on NVIDIA GPUs via the CuPy library.

Table of Contents
The ACA Algorithm: A New Paradigm

The Algorithm Cycle

ELI5: How to Solve Your Own Problem

Installation and Usage

License

The ACA Algorithm: A New Paradigm
The ACA solver is a decentralized, adaptive system that synthesizes the most successful concepts from a long series of computational experiments. It is built on four key pillars:

Pillar 1: Multi-Replica Framework
Like Parallel Tempering, ACA uses multiple copies (replicas) of the system at different temperatures. This allows the solver to explore the solution space on multiple levels simultaneously.

Hot Replicas: Act as "scouts," broadly exploring the global energy landscape and preventing the system from getting trapped early.

Cold Replicas: Act as "miners," carefully refining solutions in promising low-energy regions.

Pillar 2: Collective Learning (Entanglement)
This is the breakthrough from our earlier Entangled Replica Dynamics (ERD) experiment. The replicas are not independent. They periodically share information to build a global Correlation Matrix (C). This matrix represents the learned, collective wisdom of the entire ensemble, identifying which pairs of spins tend to be aligned in the most promising solutions discovered so far.

Pillar 3: Adaptive Non-Local Moves
This is the core of ACA's intelligence. The solver uses the global knowledge in the Correlation Matrix to propose powerful, non-local "block-flips"—flipping entire clusters of correlated spins at once. Crucially, the process is adaptive:

Each replica learns its own correlation_threshold parameter.

Based on the success rate of its recent moves, a replica can become more aggressive (lowering its threshold to propose larger, bolder flips) or more conservative (raising its threshold for smaller, refining flips).

This creates a diverse team of specialists, each tailoring its strategy to its specific temperature and region of the solution space.

Pillar 4: The Stochastic Safety Net (Replica Exchange)
This is the classic, proven mechanism that ensures robustness. The failures of our more deterministic experimental solvers (ART, SFA) taught us that a system can become trapped by its own consensus. ACA prevents this by reintroducing the replica-exchange swap move. A cold replica that is "stuck" in a local minimum can instantly trade places with a hot, high-energy replica, completely rejuvenating its search from a new, diverse starting point.

The Algorithm Cycle
The solver operates in a continuous loop, with each cycle consisting of these phases:

Local Evolution: All replicas run standard Metropolis sweeps for local refinement.

Collective Learning: The global Correlation Matrix C is updated with the latest findings from all replicas.

Adaptive Moves: Replicas perform entangled block-flips using their personal, learned thresholds.

Feedback & Adaptation: Move acceptance rates are measured, and the thresholds are adjusted.

Replica Exchange: Swap moves are attempted between adjacent replicas to prevent the system from getting trapped.

The process repeats.

ELI5: How to Solve Your Own Problem
What's an Ising Problem?
Imagine you have a huge box of tiny, powerful magnets.

Some magnets want to point UP, others want to point DOWN.

They all push and pull on each other. Some pairs want to point in the SAME direction, and other pairs want to point in OPPOSITE directions.

An Ising problem is about finding the one arrangement of all the magnets that makes the whole system the most stable and calm (the lowest possible energy). This is extremely hard because flipping one magnet can send ripples through the whole system.

What You Need
A Computer with an NVIDIA GPU.

Python and the CuPy library installed.

Your Problem, defined by two simple things:

A J matrix: A square grid of numbers. This is your "instruction manual" that says how strongly each magnet pushes or pulls on every other magnet.

An h vector: A list of numbers. This tells you if each magnet has its own personal preference to point up (+) or down (-).

How to Use the Solver
Download the aca_solver.py script.

Open the file and scroll to the very bottom, to the if __name__ == '__main__': section.

Replace the example J and h variables with your own problem's data (as NumPy arrays).

Run the script from your terminal: python aca_solver.py.

The solver will use your GPU to find the best arrangement. It will print its progress and, at the end, give you the final lowest energy it found and the best arrangement of your magnets (a list of +1s and -1s).

Installation and Usage
1. Prerequisites
An NVIDIA GPU with a modern driver.

The NVIDIA CUDA Toolkit (version 11.x or 12.x).

2. Installation
Install Python and the required libraries. Make sure to match the CuPy version to your installed CUDA Toolkit version.

# Install NumPy
pip install numpy

# Install CuPy (example for CUDA 12.x)
pip install cupy-cuda12x

3. Running the Code
Save the aca_solver.py script and execute it from your terminal.

python aca_solver.py

The script includes a self-contained example and will run automatically.

License
This project is licensed under the MIT License.
