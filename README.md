# CSE443: Constrained DNA Data Storage Autoencoder

An end-to-end differentiable deep learning framework for translating binary digital payloads into biochemical-compliant DNA sequences ({A, C, G, T}).

## Pipeline Overview
- **Differentiable Channel:** Gumbel-Softmax continuous relaxation with simulated substitution noise.
- **Biochemical Loss Constraints:** Differentiable penalties for GC-content window adherence (40%–60%) and homopolymer run suppression (max run < 3).
- **Architecture:** Joint Source-Channel Coding (JSCC) autoencoder mapping 128-bit blocks to 80-nucleotide sequences.

## Performance
- **Effective Code Rate:** 1.62 bits/nucleotide
- **Bit Error Rate (BER):** < 0.005
- **Biological Policy Violation Rate:** Reduced from 34.2% to 2.1%

## Repository Structure
- `data_generator.py`: Generates uniform binary and structured ASCII payloads.
- `model.py`: Neural autoencoder encoder-decoder architecture.
- `noise_channel.py`: Continuous Gumbel-Softmax biological channel.
- `loss.py`: Multi-objective loss with GC and homopolymer penalty terms.
- `train.py`: Training pipeline.
- `evaluate.py`: Evaluation benchmarks across BER, policy violations, and code rate.
