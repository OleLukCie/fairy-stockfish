#!/usr/bin/env python3
"""
E2: Output Scale Sweep Experiment
Multiply atomic NNUE output weights by scale factors {0.5, 1.5, 2.0}
Original = 1.0 (baseline)

Usage:
    python e2_scale_sweep.py <input_atomic.nnue> <output_dir>

Generates:
    - atomic_scale0.5.nnue
    - atomic_scale1.5.nnue  
    - atomic_scale2.0.nnue
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


def modify_output_scale(nnue_info, scale):
    """Multiply output weights by scale factor."""
    data = bytearray(nnue_info['raw_data'])
    idx = 0

    # Skip FT biases
    idx += nnue_info['l1'] * 2

    # Skip FT weights
    idx += nnue_info['ft_in'] * nnue_info['l1'] * 2

    # Skip PSQT weights
    if nnue_info['psqt_buckets'] > 0:
        idx += nnue_info['ft_in'] * nnue_info['psqt_buckets'] * 4

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

        # Skip output bias
        old_bias = struct.unpack('<i', data[idx:idx+4])[0]
        idx += 4

        # Modify output weights (int8, padded)
        out_in_pad = ((nnue_info['l3'] + 31) // 32) * 32
        print(f"  Layer Stack {ls}: Scaling {nnue_info['l3']} output weights by {scale}")
        for i in range(nnue_info['l3']):
            old_w = struct.unpack('<b', data[idx + i:idx + i + 1])[0]
            new_w = max(-127, min(127, int(round(old_w * scale))))
            data[idx + i] = struct.pack('<b', new_w)[0]
        idx += out_in_pad

    return bytes(data)


def write_nnue(output_path, nnue_info, modified_data):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', nnue_info['version']))
        f.write(struct.pack('<I', nnue_info['hash_val']))
        f.write(struct.pack('<I', len(nnue_info['description'])))
        f.write(nnue_info['description'])
        f.write(struct.pack('<I', nnue_info['ft_hash']))
        f.write(modified_data)
    print(f"  Written: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python e2_scale_sweep.py <input_atomic.nnue> <output_dir>")
        print("\n  Generates 3 scaled variants:")
        print("    - atomic_scale0.5.nnue  (compressed)")
        print("    - atomic_scale1.5.nnue  (amplified)")
        print("    - atomic_scale2.0.nnue  (amplified)")
        print("\n  Prediction: If scale2.0 performs best,")
        print("              the problem is evaluation amplitude.")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  E2: Output Scale Sweep Experiment")
    print("=" * 70)
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_dir}")

    print("\n  Parsing NNUE...")
    nnue_info = parse_nnue(input_path)
    print(f"  Architecture: FT({nnue_info['ft_in']}x{nnue_info['l1']}) -> L1 -> L2 -> Out")

    scales = [0.5, 1.5, 2.0]

    for scale in scales:
        print(f"\n  --- Scale = {scale} ---")
        modified_data = modify_output_scale(nnue_info, scale)
        output_name = f"atomic_scale{scale}.nnue"
        output_path = output_dir / output_name
        write_nnue(output_path, nnue_info, modified_data)

    print("\n" + "=" * 70)
    print("  E2 complete. Test all 3 models in atomic mode.")
    print("  Compare against Original atomic (scale=1.0).")


if __name__ == "__main__":
    main()
