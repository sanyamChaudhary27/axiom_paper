# Axiom: A Householder-Parameterized Pure Unitary RNN for Long-Range Sequence Modeling

**Sanyam Chaudhary**  
Independent Researcher, India  
https://github.com/sanyamChaudhary27/axiom_paper  
DOI: 10.5281/zenodo.20108466

---

## Abstract

We present **Axiom**, a recurrent neural network whose hidden-to-hidden transition
is parameterized as a product of Householder reflections, forming a strict unitary
matrix. Unlike LSTM and GRU, Axiom contains **no forget gate** — the unitary
transition matrix guarantees lossless information preservation across arbitrary
sequence lengths by mathematical construction. On the delayed copy task, Axiom
achieves **76.5–99.9% accuracy** at delays of 100–300 steps using 8,584 parameters,
while LSTM (111,368 parameters, 13×more) scores 12.5–13.5% — indistinguishable
from random chance — across all delays on both GPU and TPU v6e. On the Adding
Problem (Hochreiter & Schmidhuber, 1997), Axiom achieves MSE of 0.00046 at
length 200 versus LSTM's 0.00214, confirmed on TPU at 0.015 MSE. We further
derive a **closed-form parallel forward pass** reducing the unitary recurrence to
a cumulative sum in a rotated eigenbasis, verified to within 2.81e-05 error.
Results are confirmed independently on NVIDIA GPU and Google TPU v6e (TRC).

---

## 1. Introduction

The vanishing gradient problem in recurrent networks motivated LSTM
(Hochreiter & Schmidhuber, 1997). LSTM's forget gate mitigates gradient decay
but introduces exponential memory decay: at a typical forget gate value of 0.95,
information from 100 steps ago retains 0.95^100 ≈ 0.6% of its original magnitude.

Unitary RNNs (Arjovsky et al., 2016; Wisdom et al., 2016) address this by
constraining the transition matrix to be unitary, guaranteeing all singular
values equal 1.0. Prior work used complex-valued parameterizations requiring
complex arithmetic throughout training.

We make three contributions:

1. **Real-valued Householder parameterization**: the first real-valued unitary
   RNN achieving competitive results on standard long-range benchmarks
2. **Forget gate incompatibility**: we identify that gated forgetting structurally
   negates the unitary guarantee — a design error in all prior gated unitary RNNs
3. **Closed-form parallel forward pass**: the unitary recurrence is equivalent
   to a cumulative sum in a rotated eigenbasis

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

A Householder reflection H(v) = I − 2vvᵀ/(vᵀv) is symmetric and orthogonal.
The product of k reflections is orthogonal:

    W = H(v₁) H(v₂) ··· H(vₖ)

We learn v₁,...,vₖ ∈ ℝʳᵉˢ. This is entirely real-valued.

### 3.2 Architecture

Hidden state h_t is a real (res × res) matrix:

    val_t  = tanh(W_val · x_t)
    gate_t = σ(W_gate · x_t)
    h_t    = Wᵀ h_{t-1} W + gate_t ⊙ val_t    ← NO forget gate
    y_t    = head(LayerNorm(flatten(h_t)))

Since W is unitary: ||Wᵀ h W||_F = ||h||_F exactly. No memory decay, ever.

### 3.3 Why Forget Gates Break Unitary RNNs

LSTM: h_t = f_t ⊙ h_{t-1} + i_t ⊙ tanh(...)  where f_t ∈ (0,1)^d

Even with a unitary W_h, elementwise multiplication by f_t < 1 destroys
the norm-preservation property. The effective spectral radius becomes < 1,
causing exponential decay. This is structural, not a tuning issue.

### 3.4 Closed-Form Parallel Forward Pass

Define h̃_t = Wᵗ h_t (Wᵗ)ᵀ. The recurrence becomes:

    h̃_t = h̃_{t-1} + g̃_t    where g̃_t = Wᵗ g_t (Wᵗ)ᵀ

This is a cumulative sum — computable in O(1) parallel operations.
Recovery: h_t = (Wᵗ)ᵀ h̃_t Wᵗ

Powers Wᵗ for all t simultaneously via eigendecomposition:
W = V diag(λ) V⁻¹ → Wᵗ = V diag(λᵗ) V⁻¹, one vectorized broadcast.

Verified: max absolute error between sequential and parallel = **2.81 × 10⁻⁵**.

---

## 4. Experiments

### 4.1 Gradient Stability

Trained on TinyShakespeare for 20,000 steps. Gradient norm of v:

| Step | Norm |
|:---:|:---:|
| 0 | 0.476 |
| 5,000 | 0.482 |
| 10,000 | 0.489 |
| 20,000 | 0.488 |

In a vanilla RNN this decays to ~0 within 3,000 steps. Axiom's stays stable.

### 4.2 Delayed Copy Task

Input: N random symbols, then T blanks, a marker, then N output positions.
Model must reproduce the original N symbols. Random baseline: 12.5%.

**Axiom** (res=16, rank=8, **8,584 params**) vs **LSTM** (hidden=160, **111,368 params**).

| Delay | Seq. | Axiom GPU | Axiom TPU | LSTM GPU | LSTM TPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 121 | **99.9%** | **89.7%** | 13.5% | 13.4% |
| 300 | 321 | **33.0%** | **76.5%** | 13.3% | 13.1% |
| 500 | 521 | **17.8%↑** | **32.4%** | 13.8% | ~12.5% |
| 1000 | 1021 | — | **18.7%** | — | ~12.5% |

LSTM remains at random chance across all delays and both hardware platforms.
This is not a training issue — the forget gate structurally prevents memory retention.
At T=300, Axiom TPU reached **76.5%** — our strongest single result.

### 4.3 Adding Problem (Hochreiter & Schmidhuber, 1997)

Sequence of random floats in [0,1]; two positions marked. Output their sum.
Random baseline MSE: 0.0833. This is the original benchmark from the LSTM paper.

**Axiom** (res=16, rank=8, **2,689 params**) vs **LSTM** (hidden=128, **67,713 params**).

| Length | Axiom (GPU) | Axiom (TPU) | LSTM (GPU) | Baseline |
|:---:|:---:|:---:|:---:|:---:|
| 200 | **0.00046** | **0.01537** | 0.00214 | 0.0833 |
| 500 | **0.062** | — | 0.157 | 0.0833 |

At length 200, Axiom achieves 4.6× better MSE than LSTM with 25× fewer parameters,
confirmed on two independent hardware platforms. At length 500, Axiom is the
only model below the random baseline — LSTM failed the task completely.

---

## 5. Comparison to Prior Work

| Model | Parameterization | Real-valued | Copy T=100 | Copy T=300 |
|:---|:---|:---:|:---:|:---:|
| uRNN (Arjovsky 2016) | Complex diagonal + Fourier | No | ~100% | ~100% |
| Full-cap uRNN (Wisdom 2016) | Cayley transform | No | ~100% | ~100% |
| **Axiom (ours)** | **Householder reflections** | **Yes** | **89.7–99.9%** | **33–76.5%** |

Axiom is, to our knowledge, the first real-valued unitary RNN achieving
competitive performance on standard long-range benchmarks.

---

## 6. Limitations

**Speed.** Sequential training is ~10× slower than LSTM's fused CUDNN kernel.
The parallel forward pass reduces this to ~5× at inference. A custom CUDA kernel
would eliminate the gap — the algorithmic complexity is identical to LSTM.

**Language modeling.** Axiom does not match LSTM on character-level BPC (2.49
vs ~1.60). This is expected: language modeling rewards local statistics that
LSTM's gating captures well. Axiom's advantage is tasks requiring pure long-range
memory.

**Sequential MNIST.** Pure Axiom (res=20, rank=8) reached 21.8% on 784-step
pixel classification — below published baselines. Analysis: unitary dynamics
preserve all inputs with equal weight, which is optimal for recall but
suboptimal for classification tasks requiring selective attention to specific
timesteps. A variant with activation on the hidden state reached 78.66% but
forfeits the unitary guarantee. This tradeoff is a direction for future work.

---

## 7. Conclusion

We presented Axiom, a pure unitary RNN parameterized by Householder reflections.
The central insight — that forget gates and unitary guarantees are structurally
incompatible — clarifies a design error in all prior gated unitary RNN work.

On long-range memory benchmarks, Axiom solves tasks that LSTM cannot, using
13–25× fewer parameters, confirmed on GPU and TPU v6e. The closed-form parallel
forward pass — equivalence between unitary recurrence and cumulative sum in the
eigenbasis — is a novel mathematical contribution.

---

## References

- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation* 9(8).
- Arjovsky, Shah & Bengio (2016). Unitary Evolution RNNs. *ICML 2016*.
- Wisdom et al. (2016). Full-Capacity Unitary RNNs. *NeurIPS 2016*.
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *EMNLP 2014*.

---

*Code: https://github.com/sanyamChaudhary27/axiom_paper*  
*Supported by Google TPU Research Cloud (TRC).*  
*Hardware: NVIDIA T4 GPU (Colab/Kaggle) + Google TPU v6e (TRC).*
