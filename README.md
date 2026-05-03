# Axiom: A Householder-Parameterized Pure Unitary RNN

**Sanyam Chaudhary** · Independent Researcher, India

> *A recurrent neural network with provably lossless long-range memory,
> parameterized by real-valued Householder reflections.*

---

## Key Results

| Task | Axiom | LSTM | Axiom params | LSTM params |
|:---|:---:|:---:|:---:|:---:|
| Delayed Copy (T=100) | **99.9%** | 13.5% (random) | 8,584 | 111,368 |
| Delayed Copy (T=300) | **33.0%** | 13.3% (random) | 8,584 | 111,368 |
| Adding Problem (L=200) | **0.00046 MSE** | 0.00214 MSE | 2,689 | 67,713 |
| Adding Problem (L=500) | **0.062 MSE** | 0.157 MSE | 2,689 | 67,713 |

LSTM failed completely on the copy task at all delay lengths — it never exceeded
random chance (12.5%). Axiom solved T=100 at 99.9% with 13× fewer parameters.

At L=500 on the Adding Problem (LSTM's own 1997 benchmark), Axiom is the only
model below the random baseline MSE of 0.0833. LSTM failed completely.

---

## Why It Works

Standard RNNs suffer from vanishing gradients. LSTM uses forget gates to help —
but forget gates cause **exponential memory decay**:

```
forget gate = 0.95  →  after 100 steps: 0.95^100 = 0.006
                    →  after 500 steps: 0.95^500 ≈ 5e-12
```

The memory is mathematically gone.

Axiom uses a **unitary transition matrix** built from Householder reflections:

```
h_t = Wᵀ h_{t-1} W  +  gate_t * val_t      (no forget gate)
```

Since W is unitary: `||Wᵀ h W|| = ||h||` exactly. Information from step 1
is preserved with the **same magnitude at step 1,000,000**. This is a
mathematical guarantee, not a heuristic.

The key insight: **the forget gate and the unitary guarantee are incompatible.**
Axiom removes the forget gate entirely.

---

## The Math: Parallel Forward Pass

The recurrence `h_t = Wᵀ h_{t-1} W + g_t` has a closed-form parallel solution.

Define `h̃_t = Wᵗ h_t (Wᵗ)ᵀ`. Then:

```
h̃_t = h̃_{t-1} + g̃_t      (cumulative sum!)
```

This converts a sequential recurrence into `torch.cumsum` — fully parallel.
Matrix powers `Wᵗ` for all t are computed simultaneously via eigendecomposition.

Verified: sequential vs parallel agree to within **2.81e-05** max error.

---

## Architecture

```python
class AxiomRNN(nn.Module):
    """
    Hidden state: (res × res) real matrix
    Transition:   product of rank Householder reflections
    Update:       h_t = W^T @ h_{t-1} @ W  +  gate_t * val_t
    No forget gate — memory preserved by unitary guarantee.
    """
    def __init__(self, input_size, res, rank, output_size):
        # res:  hidden state is (res × res) matrix
        # rank: number of Householder reflections (complexity of W)
```

---

## Experiments

All experiments run on Google Colab / Kaggle (NVIDIA GPU).
TPU experiments (Sequential MNIST, extended copy task) in progress via Google TRC.

### Reproduce Copy Task
- copy_task.ipynb

### Reproduce Adding Problem
- adding_problem.ipynb

---

## Installation

```bash
git clone https://github.com/sanyamChaudhary27/axiom_paper
cd axiom_paper
pip install torch numpy
```

No other dependencies required. All experiments run on a single GPU.

---

## Comparison to Prior Work

| Model | Parameterization | Real-valued | Copy T=100 |
|:---|:---|:---:|:---:|
| uRNN (Arjovsky 2016) | Complex diagonal + Fourier | ❌ | ~100% |
| Full-capacity uRNN (Wisdom 2016) | Cayley transform | ❌ | ~100% |
| **Axiom (ours)** | **Householder reflections** | ✅ | **99.9%** |

Axiom is the first real-valued unitary RNN achieving competitive performance
on standard long-range benchmarks. No complex arithmetic required.

---

## Paper

[axiom_white_paper.md] — full paper with theory, experiments, and analysis.

*Supported by Google TPU Research Cloud (TRC).*

---

## Citation

```bibtex
@article{chaudhary2026axiom,
  title={Axiom: A Householder-Parameterized Pure Unitary RNN
         for Long-Range Sequence Modeling},
  author={Chaudhary, Sanyam},
  year={2026},
  url={https://github.com/sanyamChaudhary27/axiom_paper}
}
```

---

## License

MIT License. Use freely, cite if useful.
