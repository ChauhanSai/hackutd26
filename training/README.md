# Benoit: Reinforcement Learning Training Pipeline

This directory contains the complete Python training pipeline for **Benoit**, an autonomous ScrabbleBot agent that learns competitive bidding and word-playing strategies via self-play PPO.

---

## Architecture Highlights

1. **Gymnasium Environment (`environment.py`)**:
   - High-throughput headless simulation (>30,000 steps/sec per core).
   - 6-player lobby with Vickrey auctions (second-price payment, reserve 1, random tie-breaking).
   - 98-tile Scrabble bag distribution; 5 private starting tiles dealt per player.
   - Exact word scoring integer division: `(baseScore * num) // denom` with superlinear length multiplier (up to 3.0x for $\ge 7$ letters).

2. **ENABLE Anagram Solver (`dictionary.py`)**:
   - Alphabet-sorted anagram Trie over all 172,820 words in the ENABLE dictionary.
   - Microsecond lookup ($<25\,\mu\text{s}$) that finds the exact optimal playable word from any rack without checking irrelevant permutations.

3. **Neural Network Architecture (`models.py`)**:
   - **Implicit Opponent Tracking**: Ingests sequential public event history (`AuctionResult` and `WordPlay`) using a lightweight Transformer or GRU encoder to learn opponent rack and balance states without hand-crafted features.
   - **Private State Encoder**: Projects balance, rack letter counts, current auction tile, and best-available word score/len/bonus.
   - **Continuous Actor Head**:
     - `BidAmount`: Continuous ratio in $[0, 1]$ scaled to available balance.
     - `PlayWordThreshold`: Threshold dictating whether to hold letters for superlinear multiplier bonuses or play the highest-scoring word immediately.
   - **Critic Head**: Predicts expected terminal outcome for Generalized Advantage Estimation (GAE).

4. **Self-Play Curriculum & Sparse Rewards (`train.py`)**:
   - Rewards are strictly sparse: $r_t = 0$ for all non-terminal steps, with terminal reward optimizing 6-player match placement (1st place $= +1.0$, 6th place $= -1.0$).
   - Mixes rule-based baseline bots (Cheapskate, ValueBidder, Aggressive, Hoarder) with frozen snapshots of past iterations to build a robust curriculum.

5. **ONNX Export (`export_onnx.py`)**:
   - Exports trained PyTorch weights to standard ONNX format for the production Rust client.

---

## Quickstart

```bash
cd "/Users/saichauhan/Desktop/HackUTD '26/training"

# Setup virtual environment
source .venv/bin/activate

# Run test suite
python test_suite.py

# Train Benoit (e.g. 100,000 timesteps)
python train.py --timesteps 100000 --encoder transformer --save-dir checkpoints

# Export to ONNX for Rust production client
python export_onnx.py --checkpoint checkpoints/benoit_final.pt --output ../rust-client/benoit.onnx
```

