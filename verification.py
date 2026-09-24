"""
verification.py - Reproducible Simulation Suite for Quadratic File Selection
Compatible with the Gemma 4 Developer Agent Paper Track Evaluation Guidelines.
"""

import numpy as np
import time

def run_five_node_verification():
    print("--- 1. EXAMPLES VERIFICATION: FIVE-NODE PATH INSTANCE ---")
    # Matrix coefficients from Section 4.1
    J = np.array([
        [0.0, 0.8, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.6, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 0.0]
    ])
    h = np.array([0.9, 0.8, 0.1, -0.2, -0.4])
    K = 2
    mu = 7.0

    # Optimal assignment from Section 4.2: n* = [1, 1, 0, 0, 0]
    n_star = np.array([1, 1, 0, 0, 0])
    
    # Calculate objectives
    struct_affinity = sum(J[i, j] * n_star[i] * n_star[j] for i in range(5) for j in range(i + 1, 5))
    relevance_match = np.sum(h * n_star)
    H0 = -struct_affinity - relevance_match
    penalty = mu * ((np.sum(n_star) - K) ** 2)
    H_total = H0 + penalty

    print(f"• Expected Target H_mu : -2.5")
    print(f"• Evaluated Output H_mu : {H_total:.1f}")
    if np.isclose(H_total, -2.5):
        print("✅ SUCCESS: Matches Table 1 mathematically.")
    else:
        print("❌ ERROR: Mismatch with analytical value.")
    print("-" * 60)


def run_scaled_repository_test(num_nodes=100, edge_probability=0.15, K_target=15):
    print("\n--- 2. EMPIRICAL VALIDATION: 100-NODE REPOSITORY SCALING ---")
    np.random.seed(42) # Locked seed for exact reproducibility
    
    # 1. Generate repository graph structure
    adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < edge_probability).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)
    
    # 2. Simulate normalized embedding similarities
    embeddings_sim = np.random.uniform(0.1, 0.9, size=(num_nodes, num_nodes))
    embeddings_sim = (embeddings_sim + embeddings_sim.T) / 2
    
    # 3. Formulate structural affinity matrix J
    J = 0.5 * (adjacency_matrix + adjacency_matrix.T) * np.maximum(0, embeddings_sim)
    np.fill_diagonal(J, 0)
    
    # 4. Formulate task-relevance vector h
    h = np.random.uniform(-1.0, 1.0, size=num_nodes)
    max_degree = int(np.max(np.sum(adjacency_matrix > 0, axis=1)))
    mu = max_degree + 1.0 
    
    # Baseline comparison: Pick purely based on raw semantic top-K matching
    top_k_indices = np.argsort(h)[-K_target:]
    baseline_selection = np.zeros(num_nodes, dtype=int)
    baseline_selection[top_k_indices] = 1
    
    def calculate_energy(n_vec):
        struct_term = sum(J[i, j] * n_vec[i] * n_vec[j] for i in range(num_nodes) for j in range(i + 1, num_nodes))
        relevance_term = np.sum(h * n_vec)
        H0 = -struct_term - relevance_term
        penalty = mu * ((np.sum(n_vec) - K_target) ** 2)
        return H0 + penalty

    # --- Simulated Annealing Solver Engine ---
    start_time = time.time()
    current_selection = np.zeros(num_nodes, dtype=int)
    initial_indices = np.random.choice(num_nodes, size=K_target, replace=False)
    current_selection[initial_indices] = 1
    current_energy = calculate_energy(current_selection)
    
    best_selection = np.copy(current_selection)
    best_energy = current_energy
    
    T = 10.0
    cooling_rate = 0.995
    
    for step in range(2000):
        selected = np.where(current_selection == 1)[0]
        excluded = np.where(current_selection == 0)[0]
        
        idx_to_remove = np.random.choice(selected)
        idx_to_add = np.random.choice(excluded)
        
        candidate = np.copy(current_selection)
        candidate[idx_to_remove] = 0
        candidate[idx_to_add] = 1
        
        candidate_energy = calculate_energy(candidate)
        delta_E = candidate_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_selection = candidate
            current_energy = candidate_energy
            if current_energy < best_energy:
                best_selection = np.copy(current_selection)
                best_energy = current_energy
                
        T *= cooling_rate
        
    execution_time = time.time() - start_time
    baseline_energy = calculate_energy(baseline_selection)
    
    print(f"• Baseline Top-K Energy: {baseline_energy:.4f}")
    print(f"• Optimized QUBO Energy : {best_energy:.4f}")
    print(f"• Execution Frame Trace : {execution_time:.4f} seconds")
    if np.isclose(best_energy, -24.1007, atol=1e-3):
        print("✅ SUCCESS: Matches Table 3 verified values exactly (-24.1007).")
    else:
        print("⚠ WARNING: Optimization divergence detected.")
    print("=" * 60)

if __name__ == "__main__":
    run_five_node_verification()
    run_scaled_repository_test()
verification.py
