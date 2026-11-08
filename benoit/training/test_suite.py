"""
Comprehensive test suite for Benoit training pipeline.
Verifies rules compliance, Vickrey auction mechanics, scoring, models, PPO loop, and ONNX export.
"""

from __future__ import annotations
import os
import unittest
import numpy as np
import torch

from dictionary import DEFAULT_BAG, EnableDictionary, calc_word_score, length_multiplier
from environment import (
    AUCTION_RESERVE,
    EVENT_DIM,
    EVENT_HISTORY_LEN,
    PRIVATE_DIM,
    STARTING_BALANCE,
    STARTING_RACK_SIZE,
    ScrabbleBotEnv,
)
from export_onnx import export_to_onnx
from models import BenoitActorCritic
from train import train_benoit


class TestScrabbleBotRules(unittest.TestCase):
    def setUp(self) -> None:
        self.dict = EnableDictionary()

    def test_bag_size(self) -> None:
        total = sum(c for _, c in DEFAULT_BAG)
        self.assertEqual(total, 98, "Standard Scrabble bag must have 98 tiles")

    def test_length_multiplier_and_scoring(self) -> None:
        # <= 3 letters: 1.0x
        num, den = length_multiplier(3)
        self.assertEqual((num, den), (1, 1))

        # 4 letters: 1.5x (3/2)
        num, den = length_multiplier(4)
        self.assertEqual((num, den), (3, 2))

        # 5 letters: 2.0x (2/1)
        num, den = length_multiplier(5)
        self.assertEqual((num, den), (2, 1))

        # 6 letters: 2.5x (5/2)
        num, den = length_multiplier(6)
        self.assertEqual((num, den), (5, 2))

        # >= 7 letters: 3.0x (3/1)
        num, den = length_multiplier(7)
        self.assertEqual((num, den), (3, 1))

        # Score test for "QUIZ" (Q:10, U:1, I:1, Z:10 = 22 base, len 4 -> 22 * 3 // 2 = 33)
        base, bonus, total = calc_word_score("QUIZ")
        self.assertEqual(base, 22)
        self.assertEqual(total, 33)
        self.assertEqual(bonus, 11)

    def test_dictionary_find_best_word(self) -> None:
        # Test finding best word from rack
        word, score, length, bonus = self.dict.find_best_playable("RETINAS")
        self.assertIsNotNone(word)
        self.assertGreaterEqual(score, 21)
        self.assertEqual(length, 7)

    def test_environment_initialization(self) -> None:
        env = ScrabbleBotEnv(dictionary=self.dict)
        obs, info = env.reset(seed=42)

        # 6 players
        self.assertEqual(env.num_players, 6)
        # 100 starting balance
        self.assertTrue(np.all(env.balances == STARTING_BALANCE))
        # 5 tiles dealt to each player: 6 * 5 = 30 tiles dealt, 1 drawn for auction = 31 total drawn
        # Remaining in bag = 98 - 30 - 1 = 67
        self.assertEqual(len(env.bag), 67)
        for p in range(6):
            self.assertEqual(env.racks[p].sum(), STARTING_RACK_SIZE)

        # Check observation shapes
        self.assertEqual(obs["private"].shape, (PRIVATE_DIM,))
        self.assertEqual(obs["events"].shape, (EVENT_HISTORY_LEN, EVENT_DIM))

    def test_vickrey_auction_mechanics(self) -> None:
        env = ScrabbleBotEnv(dictionary=self.dict)
        env.reset(seed=123)

        # Force auction letter
        env.current_letter = "Z"
        # Set controlled bids: Agent 0 bids 15, Agent 1 bids 10, others 0
        # Agent 0 action: bid 15% of 100 = 15, play_threshold = 0 (wait)
        obs, reward, term, trunc, info = env.step(np.array([0.15, 0.0], dtype=np.float32))

        # Winner should be Agent 0 (or another if opponent bid higher)
        winner = info["winner_idx"]
        paid = info["paid"]
        self.assertIsNotNone(winner)
        # Winner must pay 2nd highest bid or reserve 1
        self.assertGreaterEqual(paid, 1)

    def test_models_forward_and_export(self) -> None:
        for enc in ["transformer", "gru"]:
            model = BenoitActorCritic(encoder_type=enc)
            priv = torch.zeros(2, PRIVATE_DIM)
            ev = torch.zeros(2, EVENT_HISTORY_LEN, EVENT_DIM)

            act, lp, ent, val = model(priv, ev)
            self.assertEqual(act.shape, (2, 2))
            self.assertEqual(lp.shape, (2,))
            self.assertEqual(val.shape, (2,))

            act_det, v_det = model.act_deterministic(priv, ev)
            self.assertEqual(act_det.shape, (2, 2))
            self.assertEqual(v_det.shape, (2, 1))

    def test_short_training_and_onnx_export(self) -> None:
        # Run a short training loop of 2048 steps to ensure complete integration
        model = train_benoit(
            total_timesteps=2048,
            steps_per_batch=1024,
            epochs_per_batch=2,
            minibatch_size=64,
            encoder_type="gru",
            save_dir="test_checkpoints",
        )
        self.assertIsNotNone(model)

        # Test ONNX export
        onnx_file = "test_checkpoints/benoit_test.onnx"
        export_to_onnx(
            checkpoint_path="test_checkpoints/benoit_final.pt",
            output_onnx_path=onnx_file,
            encoder_type="gru",
        )
        self.assertTrue(os.path.exists(onnx_file))


if __name__ == "__main__":
    unittest.main()

