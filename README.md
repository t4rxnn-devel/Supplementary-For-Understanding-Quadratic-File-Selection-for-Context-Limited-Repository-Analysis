# Quadratic File Selection for Context-Limited Repository Analysis

Official repository containing the mathematical simulation, landscape analysis, and source verification scripts submitted to the **Google - The Gemma 4 Developer Agent Paper Track**.

## 📊 Summary
This work presents a matrix-based methodology for selecting source files under structural context limits. Instead of relying on multi-turn conversational repository crawling—which consumes context tokens via trace logs—the framework maps context configuration directly to a single preprocessing **Quadratic Unconstrained Binary Optimization (QUBO)** step outside the language model conversation.

## 🚀 Getting Started

### Prerequisites
Make sure you have `numpy` installed:
```bash
pip install numpy
```

### Execution
Run the baseline consistency suite and optimization metrics directly:
```bash
python verification.py
```

## 📉 Results Brief
* **Five-Node Formulation Minimum:** Validated exactly at $H_\mu = -2.5$.
* **100-Node System Optimization Energy:** The paired-swap Simulated Annealing protocol yields a combined objective energy state of **-24.1007**, producing an empirical framework advantage of **5.2764** lower energy deficit relative to standard Top-K semantic similarity pipelines.

## 📜 License
This software and associated verification code are published under the open-source **Apache License 2.0** in compliance with the hackathon award guidelines.
