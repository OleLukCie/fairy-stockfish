#!/usr/bin/env python3
"""
E1: Bias Recentering Experiment
Modify atomic NNUE output bias from +745 to 0 (recentering)

Usage:
    python e1_bias_recenter.py <input_atomic.nnue> <output_e1.nnue>
"""

import struct
import sys
from pathlib import Path


def parse_nnue(filepath):
    """Parse NNUE and return all components."""
    with open(filepath, 'rb') as f:
        version = struct.unpack('<I', f.read(4))[0]
        hash_val = struct.unpack('<I', f.read(4))[0]
        desc_len = struct.unpack('<I', f.read(4))[0]
        description = f.read(desc_len)
        ft_hash = struct.unpack('<I', f.read(4))[0]
        remaining = f.read()

    # Auto-detect architecture
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
                            error = abs(len(remaining) - total)
                            configs.append((error, ft_in, l1, l2, l3, psqt, num_ls, total))

    configs.sort()
    error, ft_in, l1, l2, l3, psqt_buckets, num_ls, expected = configs[0]

    return {
        'version': version, 'hash_val': hash_val, 'description': description,
        'ft_hash': ft_hash, 'ft_in': ft_in, 'l1': l1, 'l2': l2, 'l3': l3,
        'psqt_buckets': psqt_buckets, 'num_ls': num_ls,
        'raw_data': remaining, 'filepath': filepath
    }


def modify_output_bias(nnue_info, new_bias):
    """Modify the output bias of the NNUE file."""
    data = bytearray(nnue_info['raw_data'])
    idx = 0

    # Skip FT biases
    idx += nnue_info['l1'] * 2

    # Skip FT weights
    idx += nnue_info['ft_in'] * nnue_info['l1'] * 2

    # Skip PSQT weights
    if nnue_info['psqt_buckets'] > 0:
        idx += nnue_info['ft_in'] * nnue_info['psqt_buckets'] * 4

    # For each layer stack, find and modify output bias
    for ls in range(nnue_info['num_ls']):
        # Skip FC hash
        idx += 4

        # Skip L1 biases
        idx += nnue_info['l2'] * 4

        # Skip L1 weights (padded)
        l1_in_pad = ((2 * nnue_info['l1'] + 31) // 32) * 32
        idx += l1_in_pad * nnue_info['l2']

        # Skip L2 biases
        idx += nnue_info['l3'] * 4

        # Skip L2 weights (padded)
        l2_in_pad = ((nnue_info['l2'] + 31) // 32) * 32
        idx += l2_in_pad * nnue_info['l3']

        # Modify output bias (int32)
        old_bias = struct.unpack('<i', data[idx:idx+4])[0]
        print(f"  Layer Stack {ls}: Old bias = {old_bias}, New bias = {new_bias}")
        data[idx:idx+4] = struct.pack('<i', new_bias)
        idx += 4

        # Skip output weights (padded)
        out_in_pad = ((nnue_info['l3'] + 31) // 32) * 32
        idx += out_in_pad * 1

    return bytes(data)


def write_nnue(output_path, nnue_info, modified_data):
    """Write modified NNUE file."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', nnue_info['version']))
        f.write(struct.pack('<I', nnue_info['hash_val']))
        f.write(struct.pack('<I', len(nnue_info['description'])))
        f.write(nnue_info['description'])
        f.write(struct.pack('<I', nnue_info['ft_hash']))
        f.write(modified_data)
    print(f"\n  Written: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size:,} bytes")


def main():
    if len(sys.argv) < 3:
        print("Usage: python e1_bias_recenter.py <input_atomic.nnue> <output_e1.nnue>")
        print("\n  This experiment recenters the atomic NNUE output bias to 0.")
        print("  Original atomic bias: +745")
        print("  Chess bias: -68")
        print("  E1 target: 0 (neutral midpoint)")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print("=" * 70)
    print("  E1: Bias Recentering Experiment")
    print("=" * 70)
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_path}")

    print("\n  Parsing NNUE...")
    nnue_info = parse_nnue(input_path)
    print(f"  Architecture: FT({nnue_info['ft_in']}x{nnue_info['l1']}) -> L1 -> L2 -> Out")
    print(f"  Layer stacks: {nnue_info['num_ls']}")

    # Recenter bias to 0
    new_bias = 0
    print(f"\n  Modifying output bias to {new_bias}...")
    modified_data = modify_output_bias(nnue_info, new_bias)

    write_nnue(output_path, nnue_info, modified_data)

    print("\n  E1 complete. Test this model in atomic mode.")
    print("  Prediction: If E1 performs better than Original atomic,")
    print("              the problem is calibration (bias offset).")


if __name__ == "__main__":
    main()
