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
<img width="1782" height="883" alt="variance_layer2" src="https://github.com/user-attachments/assets/c9360dac-2934-407a-8b10-ba1e4ae1941f" />
<img width="2682" height="740" alt="hist_PSQT_Weights" src="https://github.com/user-attachments/assets/d168e4ae-f090-4b65-9b02-6bfa6be2c0a6" />
<img width="2685" height="740" alt="hist_Out_Weights" src="https://github.com/user-attachments/assets/350a6a92-5912-4213-86cd-d2cd49b26626" />
<img width="2684" height="740" alt="hist_L2_Weights" src="https://github.com/user-attachments/assets/f87e2ec1-1232-42d3-a339-fbf8f8e73414" />
<img width="2685" height="740" alt="hist_L2_Biases" src="https://github.com/user-attachments/assets/ae11194b-0a77-48b9-ae82-f428f5e253c2" />
<img width="2683" height="740" alt="hist_L1_Weights" src="https://github.com/user-attachments/assets/7ef5a8eb-b1a0-45e7-8d23-3c1d728af8e7" />
<img width="2684" height="740" alt="hist_L1_Biases" src="https://github.com/user-attachments/assets/3d10fc0b-aaca-4b82-8266-2ec69aac81c0" />
<img width="2681" height="740" alt="hist_FT_Weights" src="https://github.com/user-attachments/assets/d1045134-0bea-4c38-9dd0-ecd8afe162c9" />
<img width="2684" height="740" alt="hist_FT_Biases" src="https://github.com/user-attachments/assets/5258b0ae-8e99-4520-9f52-84c63d2d6ce4" />
<img width="1665" height="1184" alt="diff_Out_Weights" src="https://github.com/user-attachments/assets/817ee5a1-d055-4be5-bf7c-a584dd103087" />
<img width="1665" height="1184" alt="diff_L2_Weights" src="https://github.com/user-attachments/assets/32362b97-2ba5-4750-bc07-dcf1efc99578" />
<img width="1665" height="1184" alt="diff_L1_Weights" src="https://github.com/user-attachments/assets/2e9bcb47-9b5a-415f-b92e-5ff1f8abd724" />
