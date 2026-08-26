#!/usr/bin/env python3
"""
NNUE PSQT Degradation Analyzer
Run this locally with your two .nnue files and paste the output.

Usage:
    python test.py /path/to/chess.nnue /path/to/atomic.nnue
"""

import struct
import sys
import math
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import entropy as scipy_entropy, kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Some metrics will use fallback implementations.")


# ═══════════════════════════════════════════════════════════════════════════════
# NNUE PARSER (same as nnue_analyze.py)
# ═══════════════════════════════════════════════════════════════════════════════

class NNUENetwork:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem
        self.ft_in = 0
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0
        self.psqt_buckets = 0
        self.num_ls = 0
        self.ft_biases = []
        self.ft_weights = []
        self.psqt_weights = []
        self.ls_l1_biases = []
        self.ls_l1_weights = []
        self.ls_l2_biases = []
        self.ls_l2_weights = []
        self.ls_out_biases = []
        self.ls_out_weights = []
        self._parse()

    def _parse(self):
        with open(self.filepath, 'rb') as f:
            self.version = struct.unpack('<I', f.read(4))[0]
            self.hash_val = struct.unpack('<I', f.read(4))[0]
            desc_len = struct.unpack('<I', f.read(4))[0]
            self.description = f.read(desc_len).decode('utf-8', errors='replace')
            ft_hash = struct.unpack('<I', f.read(4))[0]
            remaining = f.read()
            self._detect_and_parse(remaining)

    def _detect_and_parse(self, data):
        configs = []
        for ft_in in [41024, 45056, 52416, 768]:
            for l1 in [256, 512, 1024]:
                for l2 in [16, 32, 64]:
                    for l3 in [32, 64]:
                        for psqt in [0, 8]:
                            for num_ls in [1, 2, 4, 8]:
                                ft_b = l1 * 2
                                ft_w = ft_in * l1 * 2
                                psqt_w = ft_in * psqt * 4 if psqt > 0 else 0
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
        error, self.ft_in, self.l1, self.l2, self.l3, self.psqt_buckets, self.num_ls, expected = configs[0]
        self._parse_weights(data)

    def _parse_weights(self, data):
        idx = 0
        self.ft_biases = list(struct.unpack(f'<{self.l1}h', data[idx:idx + self.l1 * 2]))
        idx += self.l1 * 2
        ft_w_count = self.ft_in * self.l1
        self.ft_weights = list(struct.unpack(f'<{ft_w_count}h', data[idx:idx + ft_w_count * 2]))
        idx += ft_w_count * 2
        if self.psqt_buckets > 0:
            psqt_count = self.ft_in * self.psqt_buckets
            self.psqt_weights = list(struct.unpack(f'<{psqt_count}i', data[idx:idx + psqt_count * 4]))
            idx += psqt_count * 4
        for ls in range(self.num_ls):
            idx += 4
            l1_b = list(struct.unpack(f'<{self.l2}i', data[idx:idx + self.l2 * 4]))
            idx += self.l2 * 4
            l1_in_pad = ((2 * self.l1 + 31) // 32) * 32
            l1_w_count = l1_in_pad * self.l2
            l1_w = list(struct.unpack(f'<{l1_w_count}b', data[idx:idx + l1_w_count]))
            idx += l1_w_count
            l1_w = [l1_w[r * l1_in_pad + c] for r in range(self.l2) for c in range(2 * self.l1)]
            l2_b = list(struct.unpack(f'<{self.l3}i', data[idx:idx + self.l3 * 4]))
            idx += self.l3 * 4
            l2_in_pad = ((self.l2 + 31) // 32) * 32
            l2_w_count = l2_in_pad * self.l3
            l2_w = list(struct.unpack(f'<{l2_w_count}b', data[idx:idx + l2_w_count]))
            idx += l2_w_count
            l2_w = [l2_w[r * l2_in_pad + c] for r in range(self.l3) for c in range(self.l2)]
            out_b = list(struct.unpack('<i', data[idx:idx + 4]))
            idx += 4
            out_in_pad = ((self.l3 + 31) // 32) * 32
            out_w_count = out_in_pad * 1
            out_w = list(struct.unpack(f'<{out_w_count}b', data[idx:idx + out_w_count]))
            idx += out_w_count
            out_w = out_w[:self.l3]
            self.ls_l1_biases.append(l1_b)
            self.ls_l1_weights.append(l1_w)
            self.ls_l2_biases.append(l2_b)
            self.ls_l2_weights.append(l2_w)
            self.ls_out_biases.append(out_b)
            self.ls_out_weights.append(out_w)

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


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(weights, bins=256):
    """Compute Shannon entropy of weight distribution."""
    if HAS_SCIPY:
        hist, _ = np.histogram(weights, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return scipy_entropy(probs, base=2)
    else:
        min_w, max_w = min(weights), max(weights)
        if min_w == max_w:
            return 0.0
        bin_width = (max_w - min_w) / bins
        counts = [0] * bins
        for w in weights:
            idx = min(int((w - min_w) / bin_width), bins - 1)
            counts[idx] += 1
        total = len(weights)
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h


def compute_kurtosis(weights):
    """Compute excess kurtosis."""
    if HAS_SCIPY:
        return kurtosis(weights)
    else:
        n = len(weights)
        mean = sum(weights) / n
        var = sum((w - mean) ** 2 for w in weights) / n
        if var == 0:
            return float('inf')
        m4 = sum((w - mean) ** 4 for w in weights) / n
        return m4 / (var ** 2) - 3.0


def analyze_layer(name, weights, dtype=""):
    """Analyze a single weight tensor."""
    n = len(weights)
    if n == 0:
        return None

    mean = sum(weights) / n
    var = sum((w - mean) ** 2 for w in weights) / n
    std = math.sqrt(var) if var > 0 else 0
    min_w = min(weights)
    max_w = max(weights)

    s = sorted(weights)
    q25 = s[n // 4]
    q50 = s[n // 2]
    q75 = s[3 * n // 4]

    sparsity_10 = sum(1 for w in weights if abs(w) < 10) / n * 100
    sparsity_50 = sum(1 for w in weights if abs(w) < 50) / n * 100

    h = shannon_entropy(weights)
    k = compute_kurtosis(weights)

    unique = len(set(weights))
    unique_ratio = unique / n

    zero_ratio = sum(1 for w in weights if w == 0) / n * 100

    print(f"\n  [{name}] {dtype}")
    print(f"    Count:              {n:,}")
    print(f"    Mean:               {mean:>12.4f}")
    print(f"    Std:                {std:>12.4f}")
    print(f"    Min / Max:          {min_w:>12.2f} / {max_w:>12.2f}")
    print(f"    Range:              {max_w - min_w:>12.2f}")
    print(f"    Q25 / Median / Q75: {q25:>12.2f} / {q50:>12.2f} / {q75:>12.2f}")
    print(f"    Sparsity (|w|<10):  {sparsity_10:>11.2f}%")
    print(f"    Sparsity (|w|<50):  {sparsity_50:>11.2f}%")
    print(f"    Zero Ratio:         {zero_ratio:>11.2f}%")
    print(f"    Shannon Entropy:    {h:>12.4f} bits")
    print(f"    Kurtosis:           {k:>12.4f}")
    print(f"    Unique Values:      {unique:,} ({unique_ratio*100:.2f}%)")

    return {
        'name': name, 'count': n, 'mean': mean, 'std': std,
        'min': min_w, 'max': max_w, 'range': max_w - min_w,
        'median': q50, 'sparsity_10': sparsity_10, 'sparsity_50': sparsity_50,
        'zero_ratio': zero_ratio, 'entropy': h, 'kurtosis': k,
        'unique': unique, 'unique_ratio': unique_ratio
    }


def analyze_psqt_buckets(net):
    """Analyze PSQT bucket differentiation."""
    if net.psqt_buckets == 0 or len(net.psqt_weights) == 0:
        print("\n  [PSQT Buckets] No PSQT weights found.")
        return None

    ft_in = net.ft_in
    buckets = net.psqt_buckets

    w = []
    for b in range(buckets):
        start = b * ft_in
        end = (b + 1) * ft_in
        w.append(net.psqt_weights[start:end])

    print(f"\n  [PSQT Buckets] {buckets} buckets x {ft_in} features")

    bucket_stats = []
    for b in range(buckets):
        bw = w[b]
        mean = sum(bw) / len(bw)
        std = math.sqrt(sum((x - mean) ** 2 for x in bw) / len(bw))
        bucket_stats.append({
            'mean': mean, 'std': std,
            'min': min(bw), 'max': max(bw),
            'range': max(bw) - min(bw)
        })
        print(f"    Bucket {b}: mean={mean:>10.2f} std={std:>10.2f} range={bucket_stats[-1]['range']:>12.2f}")

    print("\n    Pairwise Bucket L1 Distances (mean absolute difference):")
    distances = []
    for i in range(buckets):
        for j in range(i + 1, buckets):
            d = sum(abs(w[i][k] - w[j][k]) for k in range(ft_in)) / ft_in
            distances.append(d)
            print(f"      B{i} vs B{j}: {d:>12.2f}")

    avg_dist = sum(distances) / len(distances)
    std_dist = math.sqrt(sum((d - avg_dist) ** 2 for d in distances) / len(distances))
    min_dist = min(distances)
    max_dist = max(distances)

    print(f"\n    Summary:")
    print(f"      Mean distance:  {avg_dist:>12.2f}")
    print(f"      Std distance:   {std_dist:>12.2f}")
    print(f"      Min distance:   {min_dist:>12.2f}")
    print(f"      Max distance:   {max_dist:>12.2f}")

    cv_dist = std_dist / avg_dist if avg_dist != 0 else float('inf')
    print(f"      CV of distances: {cv_dist:>11.4f}")

    return {
        'bucket_stats': bucket_stats,
        'distances': distances,
        'mean_dist': avg_dist,
        'std_dist': std_dist,
        'min_dist': min_dist,
        'max_dist': max_dist,
        'cv_dist': cv_dist
    }


def compare_psqt(chess_net, atomic_net):
    """Direct comparison of PSQT between chess and atomic."""
    if chess_net.psqt_buckets == 0 or atomic_net.psqt_buckets == 0:
        print("PSQT comparison skipped: one or both networks lack PSQT.")
        return

    print("\n" + "=" * 80)
    print("  PSQT DIRECT COMPARISON")
    print("=" * 80)

    ft_in = chess_net.ft_in
    buckets = chess_net.psqt_buckets

    diffs = [abs(chess_net.psqt_weights[i] - atomic_net.psqt_weights[i]) 
             for i in range(len(chess_net.psqt_weights))]

    print(f"\n  Element-wise absolute differences:")
    print(f"    Count:     {len(diffs):,}")
    print(f"    Mean:      {sum(diffs)/len(diffs):>12.2f}")
    print(f"    Std:       {math.sqrt(sum((d-sum(diffs)/len(diffs))**2 for d in diffs)/len(diffs)):>12.2f}")
    print(f"    Min:       {min(diffs):>12.2f}")
    print(f"    Max:       {max(diffs):>12.2f}")
    print(f"    Median:    {sorted(diffs)[len(diffs)//2]:>12.2f}")

    if HAS_SCIPY:
        c = np.array(chess_net.psqt_weights, dtype=np.float64)
        a = np.array(atomic_net.psqt_weights, dtype=np.float64)
        corr = np.corrcoef(c, a)[0, 1]
        print(f"\n  Pearson Correlation: {corr:.6f}")
        if abs(corr) < 0.3:
            print("    => VERY LOW correlation: PSQT completely rewritten")
        elif abs(corr) < 0.7:
            print("    => MODERATE correlation: significant restructuring")
        else:
            print("    => HIGH correlation: largely preserved")

    print("\n  Per-bucket Pearson Correlation:")
    for b in range(buckets):
        start = b * ft_in
        end = (b + 1) * ft_in
        cb = chess_net.psqt_weights[start:end]
        ab = atomic_net.psqt_weights[start:end]
        if HAS_SCIPY:
            corr_b = np.corrcoef(np.array(cb, dtype=np.float64), np.array(ab, dtype=np.float64))[0, 1]
            print(f"    Bucket {b}: {corr_b:.6f}")


def compare_layers(chess_net, atomic_net):
    """Compare all layers between two networks."""
    print("\n" + "=" * 80)
    print("  LAYER-BY-LAYER COMPARISON")
    print("=" * 80)

    layers = [
        ("FT Weights", chess_net.ft_weights, atomic_net.ft_weights, "int16"),
        ("FT Biases", chess_net.ft_biases, atomic_net.ft_biases, "int16"),
        ("PSQT Weights", chess_net.psqt_weights, atomic_net.psqt_weights, "int32"),
        ("L1 Weights", chess_net.l1_weights, atomic_net.l1_weights, "int8"),
        ("L1 Biases", chess_net.l1_biases, atomic_net.l1_biases, "int32"),
        ("L2 Weights", chess_net.l2_weights, atomic_net.l2_weights, "int8"),
        ("L2 Biases", chess_net.l2_biases, atomic_net.l2_biases, "int32"),
        ("Out Weights", chess_net.out_weights, atomic_net.out_weights, "int8"),
        ("Out Biases", chess_net.out_biases, atomic_net.out_biases, "int32"),
    ]

    results = {}
    for name, cw, aw, dtype in layers:
        if len(cw) != len(aw) or len(cw) == 0:
            print(f"\n  [{name}] SKIPPED (length mismatch or empty)")
            continue

        print(f"\n  [{name}] {dtype}  (n={len(cw):,})")

        diffs = [abs(aw[i] - cw[i]) for i in range(len(cw))]
        mean_diff = sum(diffs) / len(diffs)
        std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))

        chess_mean = sum(abs(x) for x in cw) / len(cw)
        atomic_mean = sum(abs(x) for x in aw) / len(aw)

        print(f"    Mean Abs Diff:      {mean_diff:>12.4f}")
        print(f"    Std Abs Diff:       {std_diff:>12.4f}")
        print(f"    Max Abs Diff:       {max(diffs):>12.2f}")
        print(f"    Chess mean |w|:     {chess_mean:>12.4f}")
        print(f"    Atomic mean |w|:    {atomic_mean:>12.4f}")
        print(f"    Relative change:    {(atomic_mean/chess_mean - 1)*100:>11.2f}%")

        if HAS_SCIPY:
            corr = np.corrcoef(np.array(cw, dtype=np.float64), np.array(aw, dtype=np.float64))[0, 1]
            print(f"    Pearson Correlation: {corr:>11.6f}")

        if HAS_SCIPY:
            c_vec = np.array(cw, dtype=np.float64)
            a_vec = np.array(aw, dtype=np.float64)
            cos_sim = np.dot(c_vec, a_vec) / (np.linalg.norm(c_vec) * np.linalg.norm(a_vec))
            print(f"    Cosine Similarity:   {cos_sim:>11.6f}")

        results[name] = {
            'mean_diff': mean_diff, 'std_diff': std_diff,
            'chess_mean_abs': chess_mean, 'atomic_mean_abs': atomic_mean
        }

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python test.py <chess.nnue> <atomic.nnue>")
        sys.exit(1)

    chess_path = Path(sys.argv[1])
    atomic_path = Path(sys.argv[2])

    print("=" * 80)
    print("  NNUE PSQT DEGRADATION ANALYZER")
    print("=" * 80)

    print(f"\nLoading: {chess_path.name}")
    chess_net = NNUENetwork(chess_path)
    print(f"  -> FT({chess_net.ft_in}x{chess_net.l1}) PSQT_buckets={chess_net.psqt_buckets}")

    print(f"\nLoading: {atomic_path.name}")
    atomic_net = NNUENetwork(atomic_path)
    print(f"  -> FT({atomic_net.ft_in}x{atomic_net.l1}) PSQT_buckets={atomic_net.psqt_buckets}")

    # Individual layer analysis
    print("\n" + "=" * 80)
    print("  CHESS NETWORK ANALYSIS")
    print("=" * 80)
    chess_results = {}
    r = analyze_layer("FT Weights", chess_net.ft_weights, "int16")
    if r: chess_results['ft_weights'] = r
    r = analyze_layer("FT Biases", chess_net.ft_biases, "int16")
    if r: chess_results['ft_biases'] = r
    r = analyze_layer("PSQT Weights", chess_net.psqt_weights, "int32")
    if r: chess_results['psqt'] = r
    r = analyze_layer("L1 Weights", chess_net.l1_weights, "int8")
    if r: chess_results['l1_weights'] = r
    r = analyze_layer("L1 Biases", chess_net.l1_biases, "int32")
    if r: chess_results['l1_biases'] = r
    r = analyze_layer("L2 Weights", chess_net.l2_weights, "int8")
    if r: chess_results['l2_weights'] = r
    r = analyze_layer("L2 Biases", chess_net.l2_biases, "int32")
    if r: chess_results['l2_biases'] = r
    r = analyze_layer("Out Weights", chess_net.out_weights, "int8")
    if r: chess_results['out_weights'] = r
    r = analyze_layer("Out Biases", chess_net.out_biases, "int32")
    if r: chess_results['out_biases'] = r

    print("\n" + "=" * 80)
    print("  ATOMIC NETWORK ANALYSIS")
    print("=" * 80)
    atomic_results = {}
    r = analyze_layer("FT Weights", atomic_net.ft_weights, "int16")
    if r: atomic_results['ft_weights'] = r
    r = analyze_layer("FT Biases", atomic_net.ft_biases, "int16")
    if r: atomic_results['ft_biases'] = r
    r = analyze_layer("PSQT Weights", atomic_net.psqt_weights, "int32")
    if r: atomic_results['psqt'] = r
    r = analyze_layer("L1 Weights", atomic_net.l1_weights, "int8")
    if r: atomic_results['l1_weights'] = r
    r = analyze_layer("L1 Biases", atomic_net.l1_biases, "int32")
    if r: atomic_results['l1_biases'] = r
    r = analyze_layer("L2 Weights", atomic_net.l2_weights, "int8")
    if r: atomic_results['l2_weights'] = r
    r = analyze_layer("L2 Biases", atomic_net.l2_biases, "int32")
    if r: atomic_results['l2_biases'] = r
    r = analyze_layer("Out Weights", atomic_net.out_weights, "int8")
    if r: atomic_results['out_weights'] = r
    r = analyze_layer("Out Biases", atomic_net.out_biases, "int32")
    if r: atomic_results['out_biases'] = r

    # PSQT Bucket Analysis
    print("\n" + "=" * 80)
    print("  CHESS PSQT BUCKET ANALYSIS")
    print("=" * 80)
    chess_psqt = analyze_psqt_buckets(chess_net)

    print("\n" + "=" * 80)
    print("  ATOMIC PSQT BUCKET ANALYSIS")
    print("=" * 80)
    atomic_psqt = analyze_psqt_buckets(atomic_net)

    # Direct PSQT comparison
    compare_psqt(chess_net, atomic_net)

    # Layer comparison
    compare_layers(chess_net, atomic_net)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if chess_psqt and atomic_psqt:
        print(f"\n  PSQT Bucket Differentiation:")
        print(f"    Chess mean bucket distance:  {chess_psqt['mean_dist']:>12.2f}")
        print(f"    Atomic mean bucket distance: {atomic_psqt['mean_dist']:>12.2f}")
        ratio = atomic_psqt['mean_dist'] / chess_psqt['mean_dist'] if chess_psqt['mean_dist'] != 0 else 0
        print(f"    Ratio (Atomic/Chess):        {ratio:>12.4f}")
        if ratio < 0.5:
            print(f"    ⚠️  Atomic PSQT bucket differentiation is severely degraded!")
        elif ratio < 0.8:
            print(f"    ⚠️  Atomic PSQT bucket differentiation is moderately degraded.")
        else:
            print(f"    ✓  PSQT bucket differentiation is largely preserved.")

    print("\n  Key Metrics Comparison:")
    print(f"    {'Layer':<20} {'Chess Entropy':>14} {'Atomic Entropy':>16} {'Ratio':>10}")
    print(f"    {'-'*20} {'-'*14} {'-'*16} {'-'*10}")
    for key in ['ft_weights', 'psqt', 'l1_weights', 'l2_weights', 'out_weights']:
        if key in chess_results and chess_results[key] and key in atomic_results and atomic_results[key]:
            ce = chess_results[key]['entropy']
            ae = atomic_results[key]['entropy']
            r = ae / ce if ce != 0 else 0
            print(f"    {key:<20} {ce:>14.4f} {ae:>16.4f} {r:>10.4f}")

    print("\n" + "=" * 80)
    print("  Analysis complete. Copy and paste this output for interpretation.")
    print("=" * 80)


if __name__ == "__main__":
    main()