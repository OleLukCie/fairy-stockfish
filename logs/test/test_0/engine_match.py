#!/usr/bin/env python3
"""
Engine vs Engine - Pure UCI, Standard Library Only
Fairy-Stockfish handles variant rules.
Python outer loop detects: 3-fold repetition & 50-move rule.

Outputs UCI-move PGN for Lichess import.
"""

import subprocess
import threading
import queue
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(r"D:\develop\chess")
ENGINE_PATH = BASE_DIR / "engine" / "fairy-stockfish-largeboard_x86-64-modern.exe"
OUTPUT_DIR = BASE_DIR / "data"

ORIGINAL_DIR = BASE_DIR / "NNUE" / "Original"
SWAPPED_DIR = BASE_DIR / "NNUE" / "Swapped"

ORIGINAL_OPTIONS = {
    "EvalFile": str(ORIGINAL_DIR / "nn-46832cfbead3.nnue"),
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

MOVE_TIME_MS = 1000
MAX_MOVES = 200

# Repetition / 50-move detection
REPETITION_THRESHOLD = 3   # 3-fold repetition = draw
MAX_NO_CAPTURE_PLY = 100   # 50 moves = 100 half-moves without capture or pawn move


# ═══════════════════════════════════════════════════════════════════════════════
# UCI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.send(f"go movetime {movetime_ms}")
        lines = self._collect("bestmove", timeout=movetime_ms / 1000 + 10)
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "(none)"
        return "(none)"

    def probe_result(self, moves):
        """Ask engine if position is terminal. Returns result or None."""
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
                    # No moves - check info for mate vs stalemate
                    for info in lines:
                        if "score mate" in info:
                            mate_num = int(info.split("score mate ")[1].split()[0])
                            if len(moves) % 2 == 1:  # Black to move, no moves
                                return "0-1" if mate_num > 0 else "1-0"
                            else:
                                return "1-0" if mate_num > 0 else "0-1"
                        if "score cp" in info:
                            return "1/2-1/2"
                    return "1/2-1/2"
        return None

    def get_fen(self, moves):
        """Get FEN from engine for position hashing."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# REPETITION & 50-MOVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DrawDetector:
    """Detects 3-fold repetition and 50-move rule from FEN strings."""

    def __init__(self):
        self.position_counts = {}  # fen_hash -> count
        self.halfmove_clock = 0     # moves since last capture or pawn move
        self.last_fen = ""

    def _hash_fen(self, fen):
        """Hash only the position part (ignore move counters for repetition)."""
        # FEN: pieces + active color + castling + en passant
        parts = fen.split()
        position_key = " ".join(parts[:4])
        return hashlib.md5(position_key.encode()).hexdigest()

    def _is_capture_or_pawn(self, fen_before, fen_after, move_uci):
        """Rough check if move is capture or pawn move."""
        if not move_uci or move_uci == "(none)":
            return False
        # Pawn move: starts from rank 2 or 7 (for white/black)
        from_sq = move_uci[:2]
        piece_before = self._get_piece_at(fen_before, from_sq)
        if piece_before and piece_before.lower() == 'p':
            return True
        # Capture: piece count decreased
        pieces_before = self._count_pieces(fen_before)
        pieces_after = self._count_pieces(fen_after)
        return pieces_after < pieces_before

    def _get_piece_at(self, fen, sq):
        """Get piece at square from FEN."""
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
        """Count total pieces on board."""
        board_fen = fen.split()[0]
        count = 0
        for c in board_fen:
            if c.isalpha():
                count += 1
        return count

    def update(self, fen_before, fen_after, move_uci):
        """Update state after a move. Returns (is_draw, reason) or (False, "")."""
        # Update 50-move clock
        if self._is_capture_or_pawn(fen_before, fen_after, move_uci):
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Check 50-move rule (75 for some variants, but use 50 as safe default)
        if self.halfmove_clock >= MAX_NO_CAPTURE_PLY:
            return True, f"50-move rule ({self.halfmove_clock} half-moves)"

        # Check 3-fold repetition
        pos_hash = self._hash_fen(fen_after)
        self.position_counts[pos_hash] = self.position_counts.get(pos_hash, 0) + 1

        if self.position_counts[pos_hash] >= REPETITION_THRESHOLD:
            return True, f"3-fold repetition (position seen {self.position_counts[pos_hash]} times)"

        return False, ""

    def reset(self):
        self.position_counts.clear()
        self.halfmove_clock = 0


# ═══════════════════════════════════════════════════════════════════════════════
# PGN
# ═══════════════════════════════════════════════════════════════════════════════

def write_pgn(white, black, variant, moves, result, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    move_text = ""
    for i, mv in enumerate(moves):
        if i % 2 == 0:
            move_text += f"{i // 2 + 1}. {mv} "
        else:
            move_text += f"{mv} "

    # Add termination reason if draw
    termination = ""
    if result == "1/2-1/2":
        termination = "\n[Termination "

    pgn = f"""[Event "Engine Match: {white} vs {black}"]
[Site "Local"]
[Date "{datetime.now().strftime('%Y.%m.%d')}"]
[Round "1"]
[White "{white}"]
[Black "{black}"]
[Result "{result}"]
[Variant "{variant}"]
[TimeControl "{MOVE_TIME_MS // 1000}s/move"]

{move_text.strip()} {result}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(pgn)
    print(f"  Saved: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MATCH
# ═══════════════════════════════════════════════════════════════════════════════

def play_match(white_engine, black_engine, variant, max_moves=200):
    moves = []
    move_num = 1
    detector = DrawDetector()

    # Get initial FEN
    fen_before = white_engine.get_fen([])
    detector.reset()
    # Record starting position
    pos_hash = detector._hash_fen(fen_before)
    detector.position_counts[pos_hash] = 1

    print(f"\n{'='*60}")
    print(f"Match: {white_engine.name} (W) vs {black_engine.name} (B)")
    print(f"Variant: {variant}")
    print(f"{'='*60}\n")

    while move_num <= max_moves:
        engine = white_engine if len(moves) % 2 == 0 else black_engine
        color = "White" if len(moves) % 2 == 0 else "Black"

        bestmove = engine.get_move(moves, MOVE_TIME_MS)

        if bestmove == "(none)" or not bestmove:
            print(f"\n  {color} has no legal moves.")
            other = black_engine if len(moves) % 2 == 0 else white_engine
            result = other.probe_result(moves)
            if result is None:
                result = "*"
            print(f"  Engine adjudication: {result}")
            break

        moves.append(bestmove)

        # Get FEN after move (from the engine that just moved)
        fen_after = engine.get_fen(moves)

        # Check draw conditions
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

        # Check engine-reported terminal
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


def run_match(white_name, white_opts, black_name, black_opts, variant, tag):
    print("\n" + "-" * 70)
    print(f"  {tag.upper()}: {variant.upper()}")
    print(f"  White: {white_name} | Black: {black_name}")
    print("-" * 70)

    w = UCIEngine(white_name, ENGINE_PATH)
    b = UCIEngine(black_name, ENGINE_PATH)
    w.start()
    b.start()
    w.init(white_opts, variant)
    b.init(black_opts, variant)

    moves, result = play_match(w, b, variant)

    write_pgn(white_name, black_name, variant.capitalize(), moves, result,
              f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")

    w.quit()
    b.quit()
    return moves, result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ENGINE vs ENGINE - Pure UCI + Draw Detection")
    print("  3-fold repetition & 50-move rule by Python")
    print("=" * 70)

    if not ENGINE_PATH.exists():
        print(f"ERROR: Engine not found: {ENGINE_PATH}")
        sys.exit(1)

    print(f"\nEngine: {ENGINE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Draw detection: 3-fold rep, 50-move rule")

    run_match("Original", ORIGINAL_OPTIONS, "Swapped", SWAPPED_OPTIONS,
              "chess", "standard")
    run_match("Original", ORIGINAL_OPTIONS, "Swapped", SWAPPED_OPTIONS,
              "atomic", "atomic")
    run_match("Swapped", SWAPPED_OPTIONS, "Original", ORIGINAL_OPTIONS,
              "chess", "standard_rev")
    run_match("Swapped", SWAPPED_OPTIONS, "Original", ORIGINAL_OPTIONS,
              "atomic", "atomic_rev")

    print("\n" + "=" * 70)
    print("  ALL MATCHES COMPLETE!")
    print(f"  PGN files: {OUTPUT_DIR}")
    print("  Import to Lichess for visualization & analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()