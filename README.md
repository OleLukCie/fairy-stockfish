### Architecture

Both networks share the identical **Fairy-Stockfish variant-nnue-pytorch** architecture:

- **FT**: 45,056 × 512 (feature transformer)

- **L1**: 1,024 × 16 (first hidden layer, 8 layer stacks)

- **L2**: 16 × 32 (second hidden layer)

- **Out**: 32 × 1 (output)

- **PSQT buckets**: 8

---

- **FT Weights** Observed: Atomic mean −9.6 vs Chess −4.1; sparsity drops ~31 % → ~22 %.

- **PSQT Weights** Observed: mean flips +71.5 → −227.2; standard deviation reduces substantially.

- **L1 Biases** Observed: mean −1010.7 → +369.7 (+1380 shift).

- **L2 Weights** Observed: sparsity increases ~20 % → ~29 %; mean shifts +2.1 → −1.8.

- **Output Bias** Observed: int16‑domain bias −68 vs +745 (813 offset).

---
