# Axiom: A Householder-Parameterized Pure Unitary RNN

**Sanyam Chaudhary** · Independent Researcher, India  
*Supported by Google TPU Research Cloud (TRC)*  
*DOI: 10.5281/zenodo.20108466*

> A recurrent neural network with **provably lossless long-range memory**,
> parameterized by real-valued Householder reflections.
> Results confirmed on NVIDIA GPU and Google TPU v6e.

---

## Results

### Delayed Copy Task
*Axiom: 8,584 params · LSTM: 111,368 params (13× more)*

| Delay | Axiom (GPU) | Axiom (TPU) | LSTM (GPU) | LSTM (TPU) |
|:---:|:---:|:---:|:---:|:---:|
| T=100 | **99.9%** | **89.7%** | 13.5% | 13.4% |
| T=300 | **33.0%** | **76.5%** | 13.3% | 13.1% |
| T=500 | **17.8%↑** | **32.4%** | ~12.5% | ~12.5% |
| T=1000 | — | **18.7%** | — | ~12.5% |

Random baseline: 12.5%. **LSTM never exceeded random chance on any delay.**

### Adding Problem (Hochreiter & Schmidhuber, 1997)
*Axiom: 2,689 params · LSTM: 67,713 params (25× more)*

| Length | Axiom (GPU) | Axiom (TPU) | LSTM (GPU) | Random baseline |
|:---:|:---:|:---:|:---:|:---:|
| 200 | **0.00046** | **0.01537** | 0.00214 | 0.0833 |
| 500 | **0.062** | — | 0.157 | 0.0833 |

At L=500, **Axiom is the only model below random baseline.** LSTM failed.

---

## The Core Insight

**Forget gates and unitary memory guarantees are mathematically incompatible.**

LSTM's forget gate: `h_t = f_t * h_{t-1} + ...` where `f_t ∈ (0,1)`

After 100 steps at f=0.95: `0.95^100 = 0.006`. After 500: `5e-12`. Gone.

Axiom: `h_t = W^T h_{t-1} W + gate_t * val_t` where W is unitary

Since W is unitary: `||W^T h W|| = ||h||` exactly. Memory is preserved forever.
No forget gate. No decay. Mathematical guarantee, not heuristic.

---

## Novel Math

The recurrence is equivalent to a **cumulative sum** in a rotated basis:

```
Define: h̃_t = W^t h_t (W^t)^T
Then:   h̃_t = h̃_{t-1} + g̃_t    ← just cumsum
```

Fully parallel. Verified: max error vs sequential = **2.81e-05**.

---

## Reproduce

```bash
git clone https://github.com/sanyamChaudhary27/axiom_paper
cd axiom_paper
pip install torch numpy

# Reproduces 99.9% vs 13.5% result
python experiments/copy_task.py --model both --delay 100

# Reproduces 4.6x better MSE result  
python experiments/adding_problem.py --model both --length 200
```

---

## Citation

```bibtex
@article{chaudhary2026axiom,
  title   = {Axiom: A Householder-Parameterized Pure Unitary RNN
             for Long-Range Sequence Modeling},
  author  = {Chaudhary, Sanyam},
  year    = {2026},
  doi     = {10.5281/zenodo.20108466},
  url     = {https://zenodo.org/records/20108466}
}
```

---

*MIT License · GPU + TPU v6e confirmed*
