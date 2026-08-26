#!/usr/bin/env python3
r"""
NNUE Network Weight Analyzer - Fairy-Stockfish Compatible
Uses the official NNUEReader format from variant-nnue-pytorch.

Fairy-Stockfish NNUE Format:
  [Header]
    version: uint32     (0x7AF32F20)
    hash: uint32
    desc_len: uint32
    description: bytes[desc_len]
  [Feature Transformer]
    ft_hash: uint32
    ft_biases: int16[L1]                    # L1=512 for variant-nnue
    ft_weights: int16[FT_IN x L1]           # e.g. 41024 x 512
    psqt_weights: int32[FT_IN x PSQT_BUCKETS]
  [Layer Stacks]  (num_ls_buckets times, usually 1)
    fc_hash: uint32
    L1 biases: int32[L2]                     # L2=16
    L1 weights: int8[padded(2*L1) x L2]      # padded to 32
    L2 biases: int32[L3]                     # L3=32
    L2 weights: int8[padded(L2) x L3]        # padded to 32
    Output biases: int32[1]
    Output weights: int8[padded(L3) x 1]    # padded to 32

Usage:
    python nnue_analyze.py
"""

import struct
import sys
import math
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# FAIRY-STOCKFISH NNUE PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class NNUENetwork:
    """Parse Fairy-Stockfish NNUE file using official format."""

    # Fairy-Stockfish variant-nnue-pytorch defaults
    VERSION = 0x7AF32F20

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem

        self.version = 0
        self.hash_val = 0
        self.description = ""

        # Dimensions (will be auto-detected)
        self.ft_in = 0          # feature transformer inputs (e.g. 41024)
        self.l1 = 0           # FT outputs / accumulator size (e.g. 512)
        self.l2 = 0           # L1 outputs (e.g. 16)
        self.l3 = 0           # L2 outputs (e.g. 32)
        self.psqt_buckets = 0  # PSQT buckets (usually 8)
        self.num_ls = 0        # number of layer stacks (usually 1)

        # Raw weights (int16/int8/int32 as stored)
        self.ft_biases = []
        self.ft_weights = []     # [ft_in x l1]
        self.psqt_weights = []   # [ft_in x psqt_buckets]

        # Per-layer-stack weights
        self.ls_l1_biases = []   # list of [l2] per stack
        self.ls_l1_weights = []  # list of [2*l1 x l2] per stack
        self.ls_l2_biases = []   # list of [l3] per stack
        self.ls_l2_weights = []  # list of [l2 x l3] per stack
        self.ls_out_biases = []  # list of [1] per stack
        self.ls_out_weights = [] # list of [l3 x 1] per stack

        self._parse()

    def _parse(self):
        with open(self.filepath, 'rb') as f:
            # Header
            self.version = struct.unpack('<I', f.read(4))[0]
            self.hash_val = struct.unpack('<I', f.read(4))[0]
            desc_len = struct.unpack('<I', f.read(4))[0]
            self.description = f.read(desc_len).decode('utf-8', errors='replace')

            print(f"  File: {self.name}")
            print(f"    Version: 0x{self.version:08X}")
            print(f"    Hash: 0x{self.hash_val:08X}")
            print(f"    Description: {self.description[:80]}...")

            # Feature Transformer hash
            ft_hash = struct.unpack('<I', f.read(4))[0]
            print(f"    FT Hash: 0x{ft_hash:08X}")

            # Read FT biases to determine L1
            # We need to know L1 first. Try common values.
            # For Fairy-Stockfish variant nets, L1 is typically 512
            # Standard SF is 256. Let's detect by trying to read.

            # Read all remaining data
            remaining = f.read()
            print(f"    Remaining data: {len(remaining):,} bytes")

            self._detect_and_parse(remaining)

    def _detect_and_parse(self, data):
        """Auto-detect architecture and parse weights."""

        # Try common Fairy-Stockfish configurations
        # L1 (accumulator size): 256 (SF) or 512 (variant)
        # L2: 16 (variant) or 32 (SF)
        # L3: 32
        # PSQT buckets: 8
        # num_ls: 1 (usually)

        configs = []
        for ft_in in [41024, 45056, 52416, 768]:
            for l1 in [256, 512, 1024]:
                for l2 in [16, 32, 64]:
                    for l3 in [32, 64]:
                        for psqt in [0, 8]:
                            for num_ls in [1, 2, 4, 8]:
                                # Calculate expected size
                                ft_b = l1 * 2
                                ft_w = ft_in * l1 * 2
                                psqt_w = ft_in * psqt * 4 if psqt > 0 else 0

                                # FC layers per stack
                                l1_b = l2 * 4
                                l1_in_padded = ((2 * l1 + 31) // 32) * 32
                                l1_w = l1_in_padded * l2 * 1
                                l2_b = l3 * 4
                                l2_in_padded = ((l2 + 31) // 32) * 32
                                l2_w = l2_in_padded * l3 * 1
                                out_b = 1 * 4
                                out_in_padded = ((l3 + 31) // 32) * 32
                                out_w = out_in_padded * 1 * 1

                                fc_per_stack = 4 + l1_b + l1_w + l2_b + l2_w + out_b + out_w
                                total = ft_b + ft_w + psqt_w + 4 + num_ls * fc_per_stack

                                error = abs(len(data) - total)
                                configs.append((error, ft_in, l1, l2, l3, psqt, num_ls, total))

        configs.sort()
        best = configs[0]
        error, self.ft_in, self.l1, self.l2, self.l3, self.psqt_buckets, self.num_ls, expected = best

        print(f"    Detected: FT({self.ft_in}x{self.l1}) -> L1({2*self.l1}x{self.l2}) -> L2({self.l2}x{self.l3}) -> Out({self.l3}x1)")
        print(f"    PSQT buckets: {self.psqt_buckets}, Layer stacks: {self.num_ls}")
        print(f"    Expected: {expected:,} bytes | Actual: {len(data):,} bytes | Diff: {error}")

        if error > 100:
            print(f"    WARNING: Large size mismatch ({error} bytes). Parse may be incorrect.")

        self._parse_weights(data)

    def _parse_weights(self, data):
        """Parse weights from byte array."""
        idx = 0

        # FT biases (int16)
        self.ft_biases = list(struct.unpack(f'<{self.l1}h', data[idx:idx + self.l1 * 2]))
        idx += self.l1 * 2

        # FT weights (int16, stored as [ft_in][l1])
        ft_w_count = self.ft_in * self.l1
        self.ft_weights = list(struct.unpack(f'<{ft_w_count}h', data[idx:idx + ft_w_count * 2]))
        idx += ft_w_count * 2

        # PSQT weights (int32, if present)
        if self.psqt_buckets > 0:
            psqt_count = self.ft_in * self.psqt_buckets
            self.psqt_weights = list(struct.unpack(f'<{psqt_count}i', data[idx:idx + psqt_count * 4]))
            idx += psqt_count * 4

        # Layer stacks
        for ls in range(self.num_ls):
            # FC hash
            idx += 4

            # L1 biases (int32)
            l1_b = list(struct.unpack(f'<{self.l2}i', data[idx:idx + self.l2 * 4]))
            idx += self.l2 * 4

            # L1 weights (int8, padded)
            l1_in_pad = ((2 * self.l1 + 31) // 32) * 32
            l1_w_count = l1_in_pad * self.l2
            l1_w = list(struct.unpack(f'<{l1_w_count}b', data[idx:idx + l1_w_count]))
            idx += l1_w_count
            # Strip padding
            l1_w = [l1_w[r * l1_in_pad + c] for r in range(self.l2) for c in range(2 * self.l1)]

            # L2 biases (int32)
            l2_b = list(struct.unpack(f'<{self.l3}i', data[idx:idx + self.l3 * 4]))
            idx += self.l3 * 4

            # L2 weights (int8, padded)
            l2_in_pad = ((self.l2 + 31) // 32) * 32
            l2_w_count = l2_in_pad * self.l3
            l2_w = list(struct.unpack(f'<{l2_w_count}b', data[idx:idx + l2_w_count]))
            idx += l2_w_count
            # Strip padding
            l2_w = [l2_w[r * l2_in_pad + c] for r in range(self.l3) for c in range(self.l2)]

            # Output biases (int32)
            out_b = list(struct.unpack('<i', data[idx:idx + 4]))
            idx += 4

            # Output weights (int8, padded)
            out_in_pad = ((self.l3 + 31) // 32) * 32
            out_w_count = out_in_pad * 1
            out_w = list(struct.unpack(f'<{out_w_count}b', data[idx:idx + out_w_count]))
            idx += out_w_count
            # Strip padding
            out_w = out_w[:self.l3]

            self.ls_l1_biases.append(l1_b)
            self.ls_l1_weights.append(l1_w)
            self.ls_l2_biases.append(l2_b)
            self.ls_l2_weights.append(l2_w)
            self.ls_out_biases.append(out_b)
            self.ls_out_weights.append(out_w)

        print(f"    Parsed: {idx:,} / {len(data):,} bytes ({idx/len(data)*100:.1f}%)")
        print(f"    Weights: FT({len(self.ft_weights):,}) PSQT({len(self.psqt_weights):,})")
        print(f"    Layer stacks: {self.num_ls}")
        for i in range(self.num_ls):
            print(f"      LS{i}: L1({len(self.ls_l1_weights[i]):,}w/{len(self.ls_l1_biases[i])}b) "
                  f"L2({len(self.ls_l2_weights[i]):,}w/{len(self.ls_l2_biases[i])}b) "
                  f"Out({len(self.ls_out_weights[i])}w/{len(self.ls_out_biases[i])}b)")

    # Flatten layer stacks for comparison (use first stack)
    @property
    def l1_weights(self):
        return self.ls_l1_weights[0] if self.ls_l1_weights else []

    @property
    def l1_biases(self):
        return self.ls_l1_biases[0] if self.ls_l1_biases else []

    @property
    def l2_weights(self):
        return self.ls_l2_weights[0] if self.ls_l2_weights else []

    @property
    def l2_biases(self):
        return self.ls_l2_biases[0] if self.ls_l2_biases else []

    @property
    def out_weights(self):
        return self.ls_out_weights[0] if self.ls_out_weights else []

    @property
    def out_biases(self):
        return self.ls_out_biases[0] if self.ls_out_biases else []

    def get_stats(self, weights):
        if not weights:
            return {}
        n = len(weights)
        mean = sum(weights) / n
        var = sum((w - mean) ** 2 for w in weights) / n
        std = math.sqrt(var)
        s = sorted(weights)
        return {
            'count': n, 'mean': mean, 'std': std,
            'min': min(weights), 'max': max(weights),
            'median': s[n // 2], 'q25': s[n // 4], 'q75': s[3 * n // 4],
            'sparsity': sum(1 for w in weights if abs(w) < 10) / n * 100,
        }

    def all_stats(self):
        return {
            'FT_weights': self.get_stats(self.ft_weights),
            'FT_biases': self.get_stats(self.ft_biases),
            'PSQT_weights': self.get_stats(self.psqt_weights),
            'L1_weights': self.get_stats(self.l1_weights),
            'L1_biases': self.get_stats(self.l1_biases),
            'L2_weights': self.get_stats(self.l2_weights),
            'L2_biases': self.get_stats(self.l2_biases),
            'Out_weights': self.get_stats(self.out_weights),
            'Out_biases': self.get_stats(self.out_biases),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(net1, net2):
    s1 = net1.all_stats()
    s2 = net2.all_stats()

    print("\n" + "=" * 100)
    print("  NETWORK STATISTICS COMPARISON")
    print(f"  Left:  {net1.name} (chess NNUE)")
    print(f"  Right: {net2.name} (atomic NNUE)")
    print("=" * 100)

    for layer in s1.keys():
        a = s1[layer]
        b = s2[layer]
        if not a or not b:
            continue

        print(f"\n  {layer}:")
        print(f"    {'Metric':<12} {'nn-46832cfbead3 (chess)':>28} {'atomic-2cf13ff256cc':>28} {'Diff':>16}")
        print(f"    {'-'*12} {'-'*28} {'-'*28} {'-'*16}")

        for m in ['count', 'mean', 'std', 'min', 'max', 'median', 'sparsity']:
            v1 = a.get(m, 0)
            v2 = b.get(m, 0)
            diff = v2 - v1

            if m == 'count':
                print(f"    {m:<12} {v1:>28} {v2:>28} {diff:>16}")
            elif m == 'sparsity':
                print(f"    {m:<12} {v1:>27.2f}% {v2:>27.2f}% {diff:>15.2f}%")
            else:
                print(f"    {m:<12} {v1:>28.4f} {v2:>28.4f} {diff:>16.4f}")


def plot_histograms(net1, net2, outdir):
    if not MATPLOTLIB_OK:
        print("  matplotlib not available")
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    layers = [
        ('FT_Weights', net1.ft_weights, net2.ft_weights, 'int16'),
        ('FT_Biases', net1.ft_biases, net2.ft_biases, 'int16'),
        ('PSQT_Weights', net1.psqt_weights, net2.psqt_weights, 'int32'),
        ('L1_Weights', net1.l1_weights, net2.l1_weights, 'int8'),
        ('L1_Biases', net1.l1_biases, net2.l1_biases, 'int32'),
        ('L2_Weights', net1.l2_weights, net2.l2_weights, 'int8'),
        ('L2_Biases', net1.l2_biases, net2.l2_biases, 'int32'),
        ('Out_Weights', net1.out_weights, net2.out_weights, 'int8'),
    ]

    for title, w1, w2, dtype in layers:
        if not w1 or not w2 or len(w1) != len(w2):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{title}: Chess NNUE vs Atomic NNUE', fontsize=13, fontweight='bold')

        all_vals = w1 + w2
        min_v, max_v = min(all_vals), max(all_vals)

        axes[0].hist(w1, bins=100, range=(min_v, max_v), alpha=0.7,
                     color='steelblue', edgecolor='black', linewidth=0.3)
        axes[0].set_title(f'{net1.name}\n(chess)', fontsize=11)
        axes[0].set_xlabel(f'Weight ({dtype})', fontsize=9)
        axes[0].set_ylabel('Frequency', fontsize=9)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=0.8)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(w2, bins=100, range=(min_v, max_v), alpha=0.7,
                     color='darkorange', edgecolor='black', linewidth=0.3)
        axes[1].set_title(f'{net2.name}\n(atomic)', fontsize=11)
        axes[1].set_xlabel(f'Weight ({dtype})', fontsize=9)
        axes[1].set_ylabel('Frequency', fontsize=9)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=0.8)
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(w1, bins=100, range=(min_v, max_v), alpha=0.5,
                     color='steelblue', label='chess', edgecolor='black', linewidth=0.3)
        axes[2].hist(w2, bins=100, range=(min_v, max_v), alpha=0.5,
                     color='darkorange', label='atomic', edgecolor='black', linewidth=0.3)
        axes[2].set_title('Overlay', fontsize=11)
        axes[2].set_xlabel(f'Weight ({dtype})', fontsize=9)
        axes[2].set_ylabel('Frequency', fontsize=9)
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=0.8)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fp = outdir / f"hist_{title}.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fp}")


def plot_heatmaps(net1, net2, outdir):
    if not MATPLOTLIB_OK:
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    layers = [
        ('L1_Weights', net1.l1_weights, net2.l1_weights, 2*net1.l1, net1.l2),
        ('L2_Weights', net1.l2_weights, net2.l2_weights, net1.l2, net1.l3),
        ('Out_Weights', net1.out_weights, net2.out_weights, net1.l3, 1),
    ]

    for title, w1, w2, rows, cols in layers:
        if len(w1) != len(w2) or len(w1) != rows * cols:
            continue

        diff = [abs(w2[i] - w1[i]) for i in range(len(w1))]
        max_d = max(diff) if diff else 1

        if rows > 64:
            step = rows // 64
            sampled = []
            s_rows = 0
            for r in range(0, rows, step):
                if r < rows:
                    sampled.extend(diff[r*cols:(r+1)*cols])
                    s_rows += 1
            matrix = [sampled[r*cols:(r+1)*cols] for r in range(s_rows)]
        else:
            matrix = [diff[r*cols:(r+1)*cols] for r in range(rows)]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, cmap='hot', aspect='auto', interpolation='nearest', vmin=0, vmax=max_d)
        ax.set_title(f'Absolute Weight Difference: {title}\nChess NNUE vs Atomic NNUE',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Output Neuron', fontsize=10)
        ax.set_ylabel('Input Neuron (sampled)' if rows > 64 else 'Input Neuron', fontsize=10)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Absolute Difference', fontsize=10)
        plt.tight_layout()
        fp = outdir / f"diff_{title}.png"
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fp}")


def plot_variance(net1, net2, outdir):
    if not MATPLOTLIB_OK:
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = net1.l2
    diffs = [[] for _ in range(n)]
    for i in range(len(net1.l2_weights)):
        neuron = i % n
        diffs[neuron].append(net2.l2_weights[i] - net1.l2_weights[i])

    variances = [sum(d*d for d in lst)/len(lst) if lst else 0 for lst in diffs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(n), variances, color='steelblue', edgecolor='black', linewidth=0.5)

    top5 = sorted(range(n), key=lambda i: variances[i], reverse=True)[:5]
    for idx in top5:
        bars[idx].set_color('crimson')

    ax.set_title('Per-Neuron Variance of Weight Differences (Layer 2)\nChess NNUE vs Atomic NNUE',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Neuron Index', fontsize=10)
    ax.set_ylabel('Variance', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for rank, idx in enumerate(top5, 1):
        ax.text(idx, variances[idx], f'#{rank}', ha='center', va='bottom',
                fontsize=8, color='crimson', fontweight='bold')

    plt.tight_layout()
    fp = outdir / "variance_layer2.png"
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fp}")
    print(f"    Top 5 changed neurons: {top5}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    base = Path(r"D:\develop\chess")
    chess_net = base / "NNUE" / "Original" / "nn-46832cfbead3.nnue"
    atomic_net = base / "NNUE" / "Original" / "atomic-2cf13ff256cc.nnue"
    out = base / "data" / "nnue_analysis"

    print("=" * 100)
    print("  NNUE NETWORK WEIGHT ANALYZER")
    print("  Comparing: Chess NNUE vs Atomic NNUE")
    print("  Using Fairy-Stockfish variant-nnue-pytorch format")
    print("=" * 100)

    for fp in [chess_net, atomic_net]:
        if not fp.exists():
            print(f"ERROR: Not found: {fp}")
            sys.exit(1)

    print(f"\nChess:  {chess_net}")
    print(f"Atomic: {atomic_net}")
    print(f"Output: {out}")

    print("\n" + "-" * 100)
    print("  PARSING CHESS NNUE")
    print("-" * 100)
    n_chess = NNUENetwork(chess_net)

    print("\n" + "-" * 100)
    print("  PARSING ATOMIC NNUE")
    print("-" * 100)
    n_atomic = NNUENetwork(atomic_net)

    print_comparison(n_chess, n_atomic)

    if MATPLOTLIB_OK:
        print("\n" + "-" * 100)
        print("  GENERATING PLOTS")
        print("-" * 100)

        print("\n  1. Weight Distribution Histograms:")
        plot_histograms(n_chess, n_atomic, out)

        print("\n  2. Weight Difference Heatmaps:")
        plot_heatmaps(n_chess, n_atomic, out)

        print("\n  3. Per-Neuron Variance:")
        plot_variance(n_chess, n_atomic, out)

        print(f"\n  All plots saved to: {out}")
    else:
        print("\n  Install matplotlib: pip install matplotlib")

    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()