## Results summary

Comparison of Wasserstein distances between Healthy and CRC samples across different distance metrics:

| Method | Preprocessing | Wasserstein H0 | Wasserstein H1 | Statistical significance (p-value, H0) |
|--------|---------------|----------------|----------------|----------------------------------------|
| Euclidean | None (raw abundances) | ~9.0 × 10⁻⁵ | ~3.3 × 10⁻⁶ | – |
| Euclidean | CLR + Aitchison | 68.08 | 1.50 | p = 1.00 (H0), p = 0.155 (H1) |
| Bray-Curtis | Normalized abundances | 0.996 | 0.037 | – |

**Key findings:**

- **CLR + Aitchison** shows the strongest separation (H0 distance = 68.1), but permutation tests with balanced groups revealed **no statistical significance** (p = 1.00 for H0, p = 0.155 for H1).
- **Bray-Curtis** gives moderate signal (H0 = 1.0, H1 = 0.04), weaker than CLR + Aitchison.
- **Raw Euclidean** distance fails to detect any meaningful topological difference (H0 ~ 10⁻⁵).
- A naive comparison without group size balancing produces a **false-positive** result for H1 (p = 0.01), demonstrating the importance of proper statistical procedures in TDA.
