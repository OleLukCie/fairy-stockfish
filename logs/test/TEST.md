### Model - Rule Mismatch Combat Experiment

- **Original: Correct Matching of Model and Rule**

- **Swapped: Complete Model - Rule Mismatch**

| Folder       | Filename                   | Actual NNUE Weight  | Standard Chess Mode | Atomic Chess Mode |
| ------------ | -------------------------- | ------------------- | ------------------- | ----------------- |
| **Original** | `nn-46832cfbead3.nnue`     | nn-46832cfbead3     | ✓                   | —                 |
|              | `atomic-2cf13ff256cc.nnue` | atomic-2cf13ff256cc | —                   | ✓                 |
| **Swapped**  | `nn-46832cfbead3.nnue`     | atomic-2cf13ff256cc | ✓                   | —                 |
|              | `atomic-2cf13ff256cc.nnue` | nn-46832cfbead3     | —                   | ✓                 |

---

### test_0

`1 s/move`

共 4 局对局：

1. STANDARD：Original(W)‑Swapped(B) → **1‑0** Original 胜
2. ATOMIC：Original(W)‑Swapped(B) → **½‑½** 平局
3. STANDARD_REV：Swapped(W)‑Original(B) → **0‑1** Original 胜
4. ATOMIC_REV：Swapped(W)‑Original(B) → **½‑½** 平局

---

### test_1

`5 s/move`

共 4 个分组，每组 10 局，共 40 局

1. Standard：Original (W)‑Swapped (B)（10 局）
- Original (白)：10 胜，0 平，0 负
- Swapped (黑)：0 胜，0 平，10 负
2. Atomic：Original (W)‑Swapped (B)（10 局）
- Original (白)：6 胜，0 平，4 负
- Swapped (黑)：4 胜，0 平，6 负
3. Standard_rev：Swapped (W)‑Original (B)（10 局）
- Swapped (白)：0 胜，0 平，10 负
- Original (黑)：10 胜，0 平，0 负
4. Atomic_rev：Swapped (W)‑Original (B)（10 局）
- Swapped (白)：8 胜，1 平，1 负
- Original (黑)：1 胜，1 平，8 负

---

### test_2

`1 s/move`

Atomic：Original (W)‑Swapped (B)（10 局）

- Original (白)：8 胜，1 平，1 负

- Swapped (黑)：1 胜，1 平，8 负

Atomic_rev：Swapped (W)‑Original (B)（10 局）

- Swapped (白)：6 胜，3 平，1 负

- Original (黑)：1 胜，3 平，6 负

---

### test_3

`30 s/move`

1. Original (W)‑Swapped (B)：共 4 局
   Original (白)：1 胜，3 平，0 负 → **2.5 分** Swapped (黑)：0 胜，3 平，1 负
2. Swapped (W)‑Original (B)：共 1 局
   Swapped (白)：0 胜，1 平，0 负 → **0.5 分** Original (黑)：0 胜，1 平，0 负

---

**对比**

| 模式        | 测试      | 时间        | Original            | Swapped             | 优势差       |
|:---------:|:-------:|:---------:|:-------------------:|:-------------------:|:---------:|
| **标准棋**   | test\_0 | 1 s/move  | 2.0/2 (100.0%)      | 0.0/2 (0.0%)        | **+2.0**  |
| **标准棋**   | test\_1 | 5 s/move  | 20.0/20 (100.0%)    | 0.0/20 (0.0%)       | **+20.0** |
| **标准棋**   | test\_3 | 30 s/move | 3.0/5 (60.0%)       | 2.0/5 (40.0%)       | **+1.0**  |
| **标准棋汇总** | —       | —         | **25.0/27 (92.6%)** | **2.0/27 (7.4%)**   | **+23.0** |
| **原子棋**   | test\_0 | 1 s/move  | 1.0/2 (50.0%)       | 1.0/2 (50.0%)       | **0.0**   |
| **原子棋**   | test\_1 | 5 s/move  | 7.5/20 (37.5%)      | 12.5/20 (62.5%)     | **-5.0**  |
| **原子棋**   | test\_2 | 1 s/move  | 11.0/20 (55.0%)     | 9.0/20 (45.0%)      | **+2.0**  |
| **原子棋汇总** | —       | —         | **19.5/42 (46.4%)** | **22.5/42 (53.6%)** | **-3.0**  |

**标准棋模型在标准棋模式下**

| 测试      | 时间控制      | 白方 (胜/平/负) | 黑方 (胜/平/负) | 合计 (胜/平/负)       | 胜率        | 和率        | 负率       |
|:-------:|:---------:|:----------:|:----------:|:----------------:|:---------:|:---------:|:--------:|
| test\_0 | 1 s/move  | 1/0/0      | 1/0/0      | **2/0/0**        | 100.0%    | 0.0%      | 0.0%     |
| test\_1 | 5 s/move  | 10/0/0     | 10/0/0     | **20/0/0**       | 100.0%    | 0.0%      | 0.0%     |
| test\_3 | 30 s/move | 1/3/0      | 0/1/0      | **1/4/0**        | 20.0%     | 80.0%     | 0.0%     |
| **汇总**  | —         | —          | —          | **23/4/0** (27局) | **85.2%** | **14.8%** | **0.0%** |

**原子棋模型在原子棋模式下**

| 测试      | 时间控制     | 白方 (胜/平/负) | 黑方 (胜/平/负) | 合计 (胜/平/负)        | 胜率        | 和率        | 负率        |
|:-------:|:--------:|:----------:|:----------:|:-----------------:|:---------:|:---------:|:---------:|
| test\_0 | 1 s/move | 0/1/0      | 0/1/0      | **0/2/0**         | 0.0%      | 100.0%    | 0.0%      |
| test\_1 | 5 s/move | 6/0/4      | 1/1/8      | **7/1/12**        | 35.0%     | 5.0%      | 60.0%     |
| test\_2 | 1 s/move | 8/1/1      | 1/3/6      | **9/4/7**         | 45.0%     | 20.0%     | 35.0%     |
| **汇总**  | —        | —          | —          | **16/7/19** (42局) | **38.1%** | **16.7%** | **45.2%** |

***在原子棋模式中，标准棋模型总体得分反而高于原子棋模型***

---

模型下载链接：[Download NNUE | Fairy-Stockfish](https://fairy-stockfish.github.io/nnue/)

NNUE 介绍：

- [Stockfish NNUE - Chess Programming Wiki](https://chessprogramming.org/Stockfish_NNUE)

- [Core Concepts | fairy-stockfish/variant-nnue-pytorch | DeepWiki](https://deepwiki.com/fairy-stockfish/variant-nnue-pytorch/3-core-concepts)

NNUE 分析：[https://github.com/fairy-stockfish/variant-nnue-pytorch/blob/master/serialize.py](Variant-nnue-pytorch)
