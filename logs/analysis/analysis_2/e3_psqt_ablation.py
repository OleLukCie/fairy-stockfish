#!/usr/bin/env python3
"""
E3: PSQT Ablation Experiment
Zero out all PSQT weights to test PSQT's contribution

Usage:
    python e3_psqt_ablation.py <input_atomic.nnue> <output_e3.nnue>
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
                            l2_in_pad = ((l2 + 31) // 32) * 32
                            l2_w = l2_in_pad * l3 * 1
                            out_b = 1 * 4
                            out_in_pad = ((l3 + 31) // 32) * 32
                            out_w = out_in_pad * 1 * 1
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


def ablate_psqt(nnue_info):
    """Zero out all PSQT weights."""
    data = bytearray(nnue_info['raw_data'])
    idx = 0

    # Skip FT biases
    idx += nnue_info['l1'] * 2

    # Skip FT weights
    idx += nnue_info['ft_in'] * nnue_info['l1'] * 2

    # Zero PSQT weights (int32)
    if nnue_info['psqt_buckets'] > 0:
        psqt_count = nnue_info['ft_in'] * nnue_info['psqt_buckets']
        print(f"  Zeroing {psqt_count:,} PSQT weights ({nnue_info['psqt_buckets']} buckets x {nnue_info['ft_in']} features)")
        for i in range(psqt_count):
            data[idx + i * 4:idx + i * 4 + 4] = struct.pack('<i', 0)
        idx += psqt_count * 4

    return bytes(data)


def write_nnue(output_path, nnue_info, modified_data):
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
        print("Usage: python e3_psqt_ablation.py <input_atomic.nnue> <output_e3.nnue>")
        print("\n  This experiment zeros out all PSQT weights.")
        print("  Prediction: If performance drops slightly,")
        print("              PSQT is a useful but non-critical component.")
        print("              If performance crashes,")
        print("              PSQT is essential for atomic evaluation.")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print("=" * 70)
    print("  E3: PSQT Ablation Experiment")
    print("=" * 70)
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_path}")

    print("\n  Parsing NNUE...")
    nnue_info = parse_nnue(input_path)
    print(f"  Architecture: FT({nnue_info['ft_in']}x{nnue_info['l1']}) -> L1 -> L2 -> Out")
    print(f"  PSQT buckets: {nnue_info['psqt_buckets']}")

    print("\n  Ablating PSQT weights...")
    modified_data = ablate_psqt(nnue_info)

    write_nnue(output_path, nnue_info, modified_data)

    print("\n  E3 complete. Test this model in atomic mode.")
    print("  Compare against Original atomic (with PSQT).")


if __name__ == "__main__":
    main()
