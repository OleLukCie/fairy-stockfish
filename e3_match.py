#!/usr/bin/env python3
r"""
E3 Match: PSQT Ablation - Test PSQT-zeroed model vs Original + Swapped
Usage:
    python e3_match.py [seconds_per_move] [games_per_matchup] [max_moves]
"""

import subprocess
import threading
import queue
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(r"D:\develop\chess")
ENGINE_PATH = BASE_DIR / "engine" / "fairy-stockfish-largeboard_x86-64-modern.exe"
OUTPUT_DIR = BASE_DIR / "data" / "E3"

E3_DIR = BASE_DIR / "NNUE" / "E3"
ORIGINAL_DIR = BASE_DIR / "NNUE" / "Original"
SWAPPED_DIR = BASE_DIR / "NNUE" / "Swapped"

E3_OPTIONS = {
    "EvalFile": str(E3_DIR / "atomic-2cf13ff256cc.nnue"),
    "VariantPath": str(ORIGINAL_DIR / "atomic-2cf13ff256cc.nnue"),
    "Hash": 256,
    "Threads": 4,
    "UCI_LimitStrength": "false",
    "Skill Level": 20,
    "Use NNUE": "true",
}

ORIGINAL_OPTIONS = {
    "EvalFile": str(ORIGINAL_DIR / "atomic-2cf13ff256cc.nnue"),
    "VariantPath": str(ORIGINAL_DIR / "atomic-2cf13ff256cc.nnue"),
    "Hash": 256,
    "Threads": 4,
    "UCI_LimitStrength": "false",
    "Skill Level": 20,
    "Use NNUE": "true",
}

SWAPPED_OPTIONS = {
    "EvalFile": str(SWAPPED_DIR / "nn-46832cfbead3.nnue"),
    "VariantPath": str(SWAPPED_DIR / "atomic-2cf13ff256cc.nnue"),
    "Hash": 256,
    "Threads": 4,
    "UCI_LimitStrength": "false",
    "Skill Level": 20,
    "Use NNUE": "true",
}

MOVE_TIME_SEC = 5.0
GAMES_PER_MATCHUP = 10
MAX_MOVES = 200
REPETITION_THRESHOLD = 3
MAX_NO_CAPTURE_PLY = 100


class UCIEngine:
    def __init__(self, name, engine_path):
        self.name = name
        self.engine_path = engine_path
        self.proc = None
        self.q = queue.Queue()
        self.reader = None

    def start(self):
        print(f"  Starting {self.name}...")
        self.proc = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8",
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self._wait("Stockfish")
        print(f"  {self.name} started")

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.q.put(line)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait(self, pattern, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = self.q.get(timeout=0.1)
                if pattern in line:
                    return line
            except queue.Empty:
                continue
        raise TimeoutError(f"Wait for '{pattern}' timed out")

    def _collect(self, end_pattern, timeout=15.0):
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = self.q.get(timeout=0.1)
                lines.append(line)
                if end_pattern in line:
                    return lines
            except queue.Empty:
                continue
        return lines

    def init(self, options, variant):
        self.send("uci")
        self._wait("uciok")
        self.send(f"setoption name UCI_Variant value {variant}")
        time.sleep(0.05)
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
            print(f"    {k} = {v}")
            time.sleep(0.03)
        self.send("isready")
        self._wait("readyok")
        print(f"  {self.name} ready ({variant})")

    def get_move(self, moves, movetime_ms):
        if moves:
            self.send(f"position startpos moves {' '.join(moves)}")
        else:
            self.send("position startpos")
        self.send(f"go movetime {int(movetime_ms)}")
        lines = self._collect("bestmove", timeout=movetime_ms / 1000 + 10)
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "(none)"
        return "(none)"

    def probe_result(self, moves):
        if moves:
            self.send(f"position startpos moves {' '.join(moves)}")
        else:
            self.send("position startpos")
        self.send("go movetime 50")
        lines = self._collect("bestmove", timeout=5.0)
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1 and parts[1] == "(none)":
                    for info in lines:
                        if "score mate" in info:
                            mate_num = int(info.split("score mate ")[1].split()[0])
                            if len(moves) % 2 == 1:
                                return "0-1" if mate_num > 0 else "1-0"
                            else:
                                return "1-0" if mate_num > 0 else "0-1"
                        if "score cp" in info:
                            return "1/2-1/2"
                    return "1/2-1/2"
        return None

    def get_fen(self, moves):
        if moves:
            self.send(f"position startpos moves {' '.join(moves)}")
        else:
            self.send("position startpos")
        self.send("d")
        lines = self._collect("Checkers:", timeout=2.0)
        for line in lines:
            if line.startswith("Fen: "):
                return line[5:].strip()
        return ""

    def quit(self):
        self.send("quit")
        try:
            self.proc.wait(timeout=2.0)
        except:
            self.proc.kill()
        print(f"  {self.name} stopped.")


class DrawDetector:
    def __init__(self):
        self.position_counts = {}
        self.halfmove_clock = 0

    def _hash_fen(self, fen):
        parts = fen.split()
        position_key = " ".join(parts[:4])
        return hashlib.md5(position_key.encode()).hexdigest()

    def _get_piece_at(self, fen, sq):
        board_fen = fen.split()[0]
        files = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
        rank = 8 - int(sq[1])
        file = files[sq[0]]
        rows = board_fen.split('/')
        if rank < 0 or rank >= len(rows):
            return None
        col = 0
        for c in rows[rank]:
            if c.isdigit():
                col += int(c)
            else:
                if col == file:
                    return c
                col += 1
        return None

    def _count_pieces(self, fen):
        board_fen = fen.split()[0]
        return sum(1 for c in board_fen if c.isalpha())

    def _is_capture_or_pawn(self, fen_before, fen_after, move_uci):
        if not move_uci or move_uci == "(none)":
            return False
        from_sq = move_uci[:2]
        piece = self._get_piece_at(fen_before, from_sq)
        if piece and piece.lower() == 'p':
            return True
        return self._count_pieces(fen_after) < self._count_pieces(fen_before)

    def update(self, fen_before, fen_after, move_uci):
        if self._is_capture_or_pawn(fen_before, fen_after, move_uci):
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.halfmove_clock >= MAX_NO_CAPTURE_PLY:
            return True, f"50-move rule ({self.halfmove_clock} half-moves)"

        pos_hash = self._hash_fen(fen_after)
        self.position_counts[pos_hash] = self.position_counts.get(pos_hash, 0) + 1

        if self.position_counts[pos_hash] >= REPETITION_THRESHOLD:
            return True, f"3-fold repetition (x{self.position_counts[pos_hash]})"

        return False, ""

    def reset(self):
        self.position_counts.clear()
        self.halfmove_clock = 0


def write_pgn(white, black, variant, moves, result, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    move_text = ""
    for i, mv in enumerate(moves):
        if i % 2 == 0:
            move_text += f"{i // 2 + 1}. {mv} "
        else:
            move_text += f"{mv} "

    pgn = f"""[Event "Engine Match: {white} vs {black}"]
[Site "Local"]
[Date "{datetime.now().strftime('%Y.%m.%d')}"]
[Round "1"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
[Variant "{variant}"]
[TimeControl "{MOVE_TIME_SEC}s/move"]

{move_text.strip()} {result}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(pgn)
    print(f"  Saved: {filepath}")


def play_match(white_engine, black_engine, variant, max_moves, movetime_ms):
    moves = []
    move_num = 1
    detector = DrawDetector()

    fen_before = white_engine.get_fen([])
    detector.reset()
    pos_hash = detector._hash_fen(fen_before)
    detector.position_counts[pos_hash] = 1

    print(f"\n{'='*60}")
    print(f"Match: {white_engine.name} (W) vs {black_engine.name} (B)")
    print(f"Variant: {variant} | Time: {movetime_ms/1000:.1f}s/move | Max: {max_moves} ply")
    print(f"{'='*60}\n")

    while move_num <= max_moves:
        engine = white_engine if len(moves) % 2 == 0 else black_engine
        color = "White" if len(moves) % 2 == 0 else "Black"

        bestmove = engine.get_move(moves, movetime_ms)

        if bestmove == "(none)" or not bestmove:
            print(f"\n  {color} has no legal moves.")
            other = black_engine if len(moves) % 2 == 0 else white_engine
            result = other.probe_result(moves)
            if result is None:
                result = "*"
            print(f"  Engine adjudication: {result}")
            break

        moves.append(bestmove)
        fen_after = engine.get_fen(moves)

        is_draw, reason = detector.update(fen_before, fen_after, bestmove)
        if is_draw:
            if len(moves) % 2 == 1:
                print()
            print(f"\n  Draw by {reason}")
            result = "1/2-1/2"
            break

        fen_before = fen_after

        if len(moves) % 2 == 1:
            print(f"  {move_num:3d}. {bestmove:8s}", end="  ")
        else:
            print(f"{bestmove:8s}")
            move_num += 1

        result = engine.probe_result(moves)
        if result is not None:
            if len(moves) % 2 == 1:
                print()
            print(f"\n  Game over: {result}")
            break
    else:
        print(f"\n  Max moves ({max_moves}) reached.")
        result = "*"

    if len(moves) % 2 == 1:
        print()

    print(f"\n  Result: {result}")
    print(f"  Total half-moves: {len(moves)}")
    return moves, result


def run_single_game(white_name, white_opts, black_name, black_opts,
                    variant, tag, game_num, max_moves, movetime_ms):
    print("\n" + "-" * 70)
    print(f"  [{tag}] Game {game_num}/{GAMES_PER_MATCHUP}")
    print(f"  {variant.upper()}: {white_name} (W) vs {black_name} (B)")
    print("-" * 70)

    w = UCIEngine(white_name, ENGINE_PATH)
    b = UCIEngine(black_name, ENGINE_PATH)
    w.start()
    b.start()
    w.init(white_opts, variant)
    b.init(black_opts, variant)

    moves, result = play_match(w, b, variant, max_moves, movetime_ms)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    write_pgn(white_name, black_name, variant.capitalize(), moves, result,
              f"{tag}_g{game_num}_{ts}.pgn")

    w.quit()
    b.quit()
    return result


def run_matchup(white_name, white_opts, black_name, black_opts,
                variant, tag, max_moves, movetime_ms):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    for i in range(1, GAMES_PER_MATCHUP + 1):
        result = run_single_game(white_name, white_opts, black_name, black_opts,
                                   variant, tag, i, max_moves, movetime_ms)
        results[result] = results.get(result, 0) + 1
    return results


def main():
    global MOVE_TIME_SEC, GAMES_PER_MATCHUP, MAX_MOVES

    if len(sys.argv) > 1:
        MOVE_TIME_SEC = float(sys.argv[1])
    if len(sys.argv) > 2:
        GAMES_PER_MATCHUP = int(sys.argv[2])
    if len(sys.argv) > 3:
        MAX_MOVES = int(sys.argv[3])

    movetime_ms = int(MOVE_TIME_SEC * 1000)

    print("=" * 70)
    print("  E3: PSQT Ablation Experiment - Atomic Matches")
    print(f"  Time: {MOVE_TIME_SEC}s/move | Games: {GAMES_PER_MATCHUP} | Max moves: {MAX_MOVES}")
    print("=" * 70)
    print("\n  Hypothesis: If performance drops slightly,")
    print("              PSQT is a useful but non-critical component.")
    print("              If performance crashes,")
    print("              PSQT is essential for atomic evaluation.")
    print("=" * 70)

    if not ENGINE_PATH.exists():
        print(f"ERROR: Engine not found: {ENGINE_PATH}")
        sys.exit(1)

    print(f"\nEngine: {ENGINE_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    all_stats = {}

    # Matchup 1: E3 (W) vs Original (B)
    stats = run_matchup("E3_NoPSQT", E3_OPTIONS, "Original", ORIGINAL_OPTIONS,
                        "atomic", "e3_vs_original", MAX_MOVES, movetime_ms)
    all_stats["E3 vs Original"] = stats

    # Matchup 2: Original (W) vs E3 (B)
    stats = run_matchup("Original", ORIGINAL_OPTIONS, "E3_NoPSQT", E3_OPTIONS,
                        "atomic", "original_vs_e3", MAX_MOVES, movetime_ms)
    all_stats["Original vs E3"] = stats

    # Matchup 3: E3 (W) vs Swapped (B)
    stats = run_matchup("E3_NoPSQT", E3_OPTIONS, "Swapped", SWAPPED_OPTIONS,
                        "atomic", "e3_vs_swapped", MAX_MOVES, movetime_ms)
    all_stats["E3 vs Swapped"] = stats

    # Matchup 4: Swapped (W) vs E3 (B)
    stats = run_matchup("Swapped", SWAPPED_OPTIONS, "E3_NoPSQT", E3_OPTIONS,
                        "atomic", "swapped_vs_e3", MAX_MOVES, movetime_ms)
    all_stats["Swapped vs E3"] = stats

    # Summary
    print("\n" + "=" * 70)
    print("  E3 SUMMARY")
    print("=" * 70)
    for matchup, stats in all_stats.items():
        total = sum(stats.values())
        w = stats.get("1-0", 0)
        b = stats.get("0-1", 0)
        d = stats.get("1/2-1/2", 0)
        u = stats.get("*", 0)
        print(f"\n  {matchup}")
        print(f"    White wins: {w}/{total}  |  Black wins: {b}/{total}  |  Draws: {d}/{total}")
        if total > 0:
            print(f"    White: {w/total*100:.1f}%  |  Black: {b/total*100:.1f}%  |  Draw: {d/total*100:.1f}%")

    # Key comparison
    print("\n" + "=" * 70)
    print("  KEY COMPARISON: E3 vs Swapped")
    print("=" * 70)
    e3_total_wins = all_stats["E3 vs Swapped"].get("1-0", 0) + all_stats["Swapped vs E3"].get("0-1", 0)
    e3_total_games = sum(all_stats["E3 vs Swapped"].values()) + sum(all_stats["Swapped vs E3"].values())
    print(f"  E3 wins vs Swapped: {e3_total_wins}/{e3_total_games} ({e3_total_wins/e3_total_games*100:.1f}%)")

    print("\n" + "=" * 70)
    print("  PSQT ABLATION IMPACT")
    print("=" * 70)

    # Compare E3 vs Original
    e3_vs_orig_wins = all_stats["E3 vs Original"].get("1-0", 0) + all_stats["Original vs E3"].get("0-1", 0)
    e3_vs_orig_games = sum(all_stats["E3 vs Original"].values()) + sum(all_stats["Original vs E3"].values())
    print(f"  E3 (no PSQT) wins vs Original (with PSQT): {e3_vs_orig_wins}/{e3_vs_orig_games} ({e3_vs_orig_wins/e3_vs_orig_games*100:.1f}%)")

    if e3_vs_orig_wins / e3_vs_orig_games < 0.3:
        print("  => PSQT is ESSENTIAL for atomic evaluation")
    elif e3_vs_orig_wins / e3_vs_orig_games < 0.45:
        print("  => PSQT is USEFUL but not critical")
    else:
        print("  => PSQT has MINIMAL impact (already auxiliary)")

    print(f"\n  PGN files: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
