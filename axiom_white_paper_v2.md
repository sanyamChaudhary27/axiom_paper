# Axiom: A Householder-Parameterized Pure Unitary RNN for Long-Range Sequence Modeling

**Sanyam Chaudhary**  
Independent Researcher, India  
https://github.com/sanyamChaudhary27/axiom_paper
DOI 10.5281/zenodo.20108466

---

## Abstract

We present **Axiom**, a recurrent neural network whose hidden-to-hidden transition
is parameterized as a product of Householder reflections, forming a strict unitary
matrix. Unlike LSTM and GRU, Axiom contains **no forget gate** — the unitary
transition matrix guarantees lossless information preservation across arbitrary
sequence lengths by mathematical construction. We demonstrate that on long-range
memory benchmarks, Axiom achieves **93.7–99.9% accuracy** on the 100-step delayed
copy task using only 8,584 parameters, while a size-matched LSTM scores 13.4–13.5%
(random chance) with 13× more parameters. Results are confirmed independently on
GPU (Colab) and TPU v6e (Google TRC). On the Adding Problem (Hochreiter &
Schmidhuber, 1997) at sequence length 200, Axiom achieves MSE of 0.00046 versus
LSTM's 0.00214 — a **4.6× improvement** with 25× fewer parameters. At length 500,
Axiom is the only model to fall below the random baseline while LSTM fails
completely. We further derive a **closed-form parallel forward pass** that reduces
the unitary recurrence to a cumulative sum in a rotated eigenbasis, verified
numerically to within 2.81e-05 error. These results position Axiom as a principled
solution for edge deployment scenarios requiring guaranteed long-range memory under
strict parameter constraints.

---

## 1. Introduction

The vanishing gradient problem in recurrent neural networks was identified by
Hochreiter (1991) and motivated the Long Short-Term Memory architecture
(Hochreiter & Schmidhuber, 1997). LSTM's gating mechanism mitigates gradient
decay but introduces a fundamental tradeoff: the forget gate, which controls
information flow, causes exponential memory decay over time. For a typical forget
gate value of 0.95, information from 100 steps ago retains only
0.95^100 ≈ 0.6% of its original magnitude. At 500 steps, this becomes
0.95^500 ≈ 5.2 × 10⁻¹². The memory is mathematically gone.

Unitary RNNs (Arjovsky et al., 2016; Wisdom et al., 2016) address this by
constraining the hidden-to-hidden transition matrix W to be unitary, guaranteeing
that all singular values equal exactly 1.0. This eliminates both vanishing and
exploding gradients by construction.

Prior unitary RNN work parameterized the unitary matrix using complex-valued
diagonal matrices and Fourier transforms — requiring complex arithmetic throughout
training. We propose an alternative: **real-valued Householder reflections**.

More importantly, we identify and correct a fundamental design error present in all
prior gated unitary RNN work: **the forget gate and the unitary guarantee are
mathematically incompatible.** Any sigmoid-gated forgetting negates the unit-norm
preservation that makes unitary matrices useful. Axiom resolves this by removing
the forget gate entirely.

Our contributions:

1. A real-valued Householder parameterization for unitary RNNs
2. Identification that forget gates negate the unitary memory guarantee
3. A closed-form parallel forward pass via eigendecomposition and change-of-basis
4. Empirical results on copy task and adding problem confirmed on two independent
   hardware platforms (GPU and TPU v6e), demonstrating Axiom solves tasks where
   LSTM completely fails using 13–25× fewer parameters

---

## 2. Background

### 2.1 The Vanishing Gradient Problem

For a recurrent network h_t = f(W h_{t-1} + U x_t), the gradient involves:

    ∂L/∂h_0 = (∏_{t=1}^{T} ∂h_t/∂h_{t-1}) · ∂L/∂h_T

For linear transitions this becomes W^T. If spectral radius < 1, this vanishes
exponentially. LSTM's gating is a heuristic fix; unitary parameterization is exact.

### 2.2 Why Forget Gates Break Unitary RNNs

LSTM's update rule: h_t = f_t ⊙ h_{t-1} + i_t ⊙ tanh(...) where f_t = σ(...) ∈ (0,1)^d.

Even if W_h is unitary, elementwise multiplication by f_t < 1 scales the hidden
state down every step. The spectral radius of the effective transition becomes
max(f_t) < 1, causing exponential decay. This is not a training issue — it is
structural.

**Axiom's design choice**: remove all forget-style gating. The only gate controls
what new information enters. Memory decay becomes impossible by construction.

---

## 3. Method

### 3.1 Householder Parameterization

A Householder reflection: H(v) = I − 2 vvᵀ / (vᵀv), v ∈ ℝʳᵉˢ

H(v) is symmetric and orthogonal. The product of k Householder reflections is
orthogonal:

    W = H(v₁) H(v₂) ··· H(vₖ)

We learn v₁, ..., vₖ ∈ ℝʳᵉˢ. Rank k controls complexity; k = res gives full
orthogonal group coverage. This is entirely real-valued — no complex arithmetic.

### 3.2 Model Architecture

The Axiom hidden state h_t is a real (res × res) matrix:

```
val_t  = tanh(W_val · x_t)           # value to write
gate_t = σ(W_gate · x_t)             # input gate (controls write)
h_t    = Wᵀ h_{t-1} W  +  gate_t ⊙ val_t    # NO forget gate
y_t    = head(LayerNorm(flatten(h_t)))
```

Since W is unitary: ||Wᵀ h_{t-1} W||_F = ||h_{t-1}||_F exactly.
Information from step 1 has identical magnitude at step T, for any T.

### 3.3 Gradient Stability (Empirical)

Training Axiom on character-level language modeling for 20,000 steps:

| Step | Gradient norm of v |
|:---:|:---:|
| 0 | 0.476 |
| 5,000 | 0.482 |
| 10,000 | 0.489 |
| 20,000 | 0.488 |

In a vanilla RNN, this decays toward zero within 1,000–3,000 steps.
The unitary parameterization prevents this by construction.

### 3.4 Closed-Form Parallel Forward Pass

The recurrence h_t = Wᵀ h_{t-1} W + g_t has a closed-form solution.

Define h̃_t = Wᵗ h_t (Wᵗ)ᵀ. Substituting:

    h̃_t = h̃_{t-1} + g̃_t    where g̃_t = Wᵗ g_t (Wᵗ)ᵀ

This is a cumulative sum: h̃_t = h̃_0 + cumsum(g̃_1, ..., g̃_t)

Recovery: h_t = (Wᵗ)ᵀ h̃_t Wᵗ

Matrix powers Wᵗ for all t simultaneously via eigendecomposition:

    W = V diag(λ) V⁻¹  →  Wᵗ = V diag(λᵗ) V⁻¹

where λᵗ for all t is one vectorized broadcast operation.

**Verification**: sequential and parallel implementations agree to within
max absolute error 2.81 × 10⁻⁵ on random inputs.

---

## 4. Experiments

All experiments run with Adam optimizer, lr=1e-3, gradient clip 1.0.
Results confirmed independently on NVIDIA T4 GPU (Google Colab/Kaggle) and
Google TPU v6e via Google TRC program.

### 4.1 Delayed Copy Task

**Task.** N random symbols from vocab-8, followed by T blank tokens, a marker,
then N output positions. Model must reproduce the original N symbols exactly.
Random baseline: 12.5% (1/8).

**Models.** Axiom (res=16, rank=8, **8,584 params**) vs LSTM
(hidden=160, **111,368 params**). LSTM has 13× more parameters.

| Delay | Seq. Length | Axiom (GPU) | Axiom (TPU) | LSTM (GPU) | LSTM (TPU) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 121 | **99.9%** | **93.7%** | 13.5% | 13.4% |
| 300 | 321 | **33.0%** | **49.7%** | 13.3% | 13.1% |
| 500 | 521 | **17.8%↑** | **30.6%** | 13.8% | ~12.5% |

LSTM scored at or below random chance (12.5%) across all delays on both
hardware platforms. This is not a tuning failure — it is structural. LSTM's
forget gate causes the input-phase information to decay exponentially during
the blank phase. At T=100 with forget gate 0.95: 0.95^100 = 0.006.

Axiom solved T=100 at 93.7–99.9% accuracy. The variation between runs reflects
random initialization and is within expected bounds for this task. At T=300,
Axiom TPU reached 49.7% (still learning at epoch 100) while LSTM remained
flat at random chance. Independent replication on two hardware platforms
strengthens the validity of the result.

### 4.2 Adding Problem (Hochreiter & Schmidhuber, 1997)

**Task.** Sequence of random floats in [0,1]; two positions marked. Model
outputs their sum. Random baseline MSE: 0.0833. This is the original benchmark
motivating LSTM's invention.

**Models.** Axiom (res=16, rank=8, **2,689 params**) vs LSTM
(hidden=128, **67,713 params**). LSTM has 25× more parameters.

| Seq. Length | Axiom MSE | LSTM MSE | Baseline MSE |
|:---:|:---:|:---:|:---:|
| 200 | **0.00046** | 0.00214 | 0.0833 |
| 500 | **0.06212** | 0.15735 | 0.0833 |

At length 200, Axiom achieves 4.6× better MSE than LSTM with 25× fewer
parameters. At length 500, Axiom is the only model below the random baseline.
LSTM's MSE of 0.157 exceeds 0.0833 — it failed to learn the task at all.

### 4.3 Language Modeling: Gradient Stability

Axiom (res=16, rank=8, 165,825 params) trained on TinyShakespeare for 50 epochs
with sequential batching and warm restarts. Best validation BPC: **2.4855**.

A parameter-matched LSTM achieves ~1.60 BPC. The gap is expected: language
modeling rewards short-range local statistics that LSTM's gating captures well.
Axiom's advantage is pure long-range memory, not local prediction.

The key finding: gradient stability across all 50 epochs without pathology,
confirming the theoretical property in practice.

---

## 5. Comparison to Prior Unitary RNN Work

| Model | Parameterization | Real-valued | Copy T=100 |
|:---|:---|:---:|:---:|
| uRNN (Arjovsky 2016) | Complex diagonal + Fourier | No | ~100% |
| Full-cap uRNN (Wisdom 2016) | Cayley transform | No | ~100% |
| **Axiom (ours)** | **Householder reflections** | **Yes** | **93.7–99.9%** |

Axiom is, to our knowledge, the first real-valued unitary RNN achieving
competitive performance on standard long-range benchmarks. The closed-form
parallel forward pass (Section 3.4) — equivalence between unitary recurrence
and cumulative sum in the eigenbasis — is novel.

---

## 6. Limitations

**Inference speed.** Axiom's sequential training loop is ~10× slower than
LSTM's fused CUDNN kernel. The parallel inference path reduces this to ~5×.
A custom CUDA kernel would eliminate this gap — the algorithmic complexity
is identical to LSTM. The speed difference is implementation overhead, not
algorithmic.

**Language modeling.** Axiom does not match LSTM on character-level BPC.
This is expected and consistent with its design: Axiom trades local gating
for provably lossless long-range memory.

**Sequential MNIST.** We attempted Sequential MNIST (784-step classification)
on both GPU and TPU. On GPU, each epoch took ~730 seconds (too slow to complete).
On TPU, PyTorch/XLA unrolls Python loops into the computation graph, creating
784 separate compiled operations that prevented efficient execution. A JAX
implementation using `lax.scan` would avoid this limitation and is planned
as future work. Based on the copy task trajectory (65.6% at epoch 50, still
climbing), we expect Axiom to reach competitive accuracy with sufficient compute.

---

## 7. Conclusion

We presented Axiom, a pure unitary RNN parameterized by Householder reflections,
with results confirmed independently on GPU and TPU v6e.

The central insight: forget gates and unitary guarantees are fundamentally
incompatible. Prior unitary RNNs retained gating mechanisms that partially
negate the property they were trying to preserve. Removing the forget gate
entirely, as in Axiom, is necessary to realize the theoretical benefit.

On long-range memory benchmarks, Axiom substantially outperforms LSTM using
13–25× fewer parameters — solving tasks that LSTM cannot solve at all, on
the exact benchmarks LSTM was designed for.

---

## References

- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen*. TU Munich.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation* 9(8).
- Arjovsky, M., Shah, A., & Bengio, Y. (2016). Unitary Evolution RNNs. *ICML 2016*.
- Wisdom, S., Powers, T., Hershey, J., Le Roux, J., & Atlas, L. (2016). Full-Capacity Unitary RNNs. *NeurIPS 2016*.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder. *EMNLP 2014*.

---

*Code: https://github.com/sanyamChaudhary27/axiom_paper*  
*Supported by Google TPU Research Cloud (TRC).*  
*Experiments: NVIDIA T4 GPU (Google Colab/Kaggle) + Google TPU v6e (TRC).*
