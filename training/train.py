"""
Self-play Reinforcement Learning training script for Benoit ScrabbleBot.
Uses PPO with Generalized Advantage Estimation (GAE) and frozen snapshot self-play curriculum.
"""

from __future__ import annotations
import argparse
import copy
import os
import random
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dictionary import EnableDictionary
from environment import (
    AggressiveOpponent,
    BaseOpponent,
    CheapskateOpponent,
    HoarderOpponent,
    ScrabbleBotEnv,
    ValueBidderOpponent,
)
from models import BenoitActorCritic


class PolicySnapshotOpponent(BaseOpponent):
    """
    Opponent powered by a frozen past checkpoint of Benoit for self-play curriculum.
    """

    def __init__(self, model: BenoitActorCritic, name: str = "SelfPlaySnapshot") -> None:
        super().__init__(name)
        self.model = model
        self.model.eval()

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        # Formulate private state tensor
        priv = np.zeros(57, dtype=np.float32)
        priv[0] = balance / 100.0
        priv[1:27] = np.array(rack, dtype=np.float32) / 5.0
        cli = ord(letter.upper()) - ord("A")
        if 0 <= cli < 26:
            priv[27 + cli] = 1.0
        _, best_s, best_l, best_b = best_word_info
        priv[53] = best_s / 50.0
        priv[54] = best_l / 10.0
        priv[55] = best_b / 30.0
        priv[56] = 0.5

        ev = np.zeros((32, 38), dtype=np.float32)

        priv_t = torch.from_numpy(priv).unsqueeze(0)
        ev_t = torch.from_numpy(ev).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.model.act_deterministic(priv_t, ev_t)
            action_np = action.squeeze(0).numpy()

        bid_frac = float(np.clip(action_np[0], 0.0, 1.0))
        bid = int(round(bid_frac * balance))
        play_thresh = float(action_np[1])
        play = best_word_info[0] is not None and play_thresh > 0.5
        return bid, play


class SnapshotPool:
    """Pool of frozen past network snapshots for self-play curriculum."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.snapshots: List[BenoitActorCritic] = []

    def add(self, model: BenoitActorCritic) -> None:
        frozen_model = copy.deepcopy(model)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False
        if len(self.snapshots) >= self.max_size:
            self.snapshots.pop(0)
        self.snapshots.append(frozen_model)

    def sample_opponent(self) -> Optional[BaseOpponent]:
        if not self.snapshots:
            return None
        model = random.choice(self.snapshots)
        return PolicySnapshotOpponent(model)


class RolloutBuffer:
    """Buffer storing trajectory rollouts for PPO updates."""

    def __init__(self) -> None:
        self.privates: List[np.ndarray] = []
        self.events: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.privates.clear()
        self.events.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(
        self,
        priv: np.ndarray,
        ev: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.privates.append(priv)
        self.events.append(ev)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE)."""
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    returns = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        next_val = last_value if t == n_steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def sample_opponents_for_match(snapshot_pool: SnapshotPool, num_needed: int = 5) -> List[BaseOpponent]:
    """Curriculum sampling: mix rule-based bots with past policy snapshots."""
    rule_pool = [
        CheapskateOpponent(),
        ValueBidderOpponent(),
        AggressiveOpponent(),
        HoarderOpponent(),
        ValueBidderOpponent(),
    ]
    opponents: List[BaseOpponent] = []
    for _ in range(num_needed):
        # 20% chance to pick snapshot if available, 80% rule-based
        if snapshot_pool.snapshots and random.random() < 0.2:
            opp = snapshot_pool.sample_opponent()
            if opp:
                opponents.append(opp)
                continue
        opponents.append(random.choice(rule_pool))
    return opponents


def train_benoit(
    total_timesteps: int = 50_000,
    steps_per_batch: int = 2048,
    epochs_per_batch: int = 4,
    minibatch_size: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    encoder_type: str = "transformer",
    save_dir: str = "checkpoints",
    snapshot_interval: int = 5,
) -> BenoitActorCritic:
    """Train Benoit using PPO with self-play curriculum."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Benoit on device: {device} | Encoder: {encoder_type}")

    shared_dict = EnableDictionary()
    env = ScrabbleBotEnv(dictionary=shared_dict, reward_type="dense")

    model = BenoitActorCritic(encoder_type=encoder_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    snapshot_pool = SnapshotPool(max_size=10)
    buffer = RolloutBuffer()

    num_iterations = total_timesteps // steps_per_batch
    step_count = 0
    recent_rewards: List[float] = []

    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        episodes_completed = 0
        wins_count = 0
        buffer.clear()
        model.eval()

        # Collect rollouts
        steps_collected = 0
        while steps_collected < steps_per_batch:
            # Set curriculum opponents for this match
            opps = sample_opponents_for_match(snapshot_pool, num_needed=env.num_players - 1)
            env.set_opponents(opps)

            obs, _ = env.reset()
            done = False

            while not done:
                priv_np = obs["private"]
                ev_np = obs["events"]

                priv_t = torch.from_numpy(priv_np).unsqueeze(0).to(device)
                ev_t = torch.from_numpy(ev_np).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_t, log_prob_t, _, value_t = model(priv_t, ev_t)
                    action_np = action_t.squeeze(0).cpu().numpy()
                    log_prob_f = float(log_prob_t.item())
                    value_f = float(value_t.item())

                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                buffer.add(
                    priv=priv_np,
                    ev=ev_np,
                    action=action_np,
                    log_prob=log_prob_f,
                    reward=reward,
                    value=value_f,
                    done=done,
                )

                obs = next_obs
                steps_collected += 1
                step_count += 1

                if done:
                    episodes_completed += 1
                    recent_rewards.append(reward)
                    if info.get("agent0_rank") == 1:
                        wins_count += 1
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)

        # Value for last step
        with torch.no_grad():
            last_priv = torch.from_numpy(obs["private"]).unsqueeze(0).to(device)
            last_ev = torch.from_numpy(obs["events"]).unsqueeze(0).to(device)
            last_val = float(model.get_value(last_priv, last_ev).item()) if not done else 0.0

        # Compute GAE
        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            last_value=last_val,
            gamma=gamma,
            lam=gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare tensors for PPO update
        privates_t = torch.tensor(np.array(buffer.privates), dtype=torch.float32, device=device)
        events_t = torch.tensor(np.array(buffer.events), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(buffer.actions), dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # PPO Update Epochs
        model.train()
        n_samples = len(buffer.privates)
        indices = np.arange(n_samples)

        clip_losses, vf_losses, ent_losses = [], [], []

        for _ in range(epochs_per_batch):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, n_samples)
                mb_idx = indices[start_idx:end_idx]

                mb_priv = privates_t[mb_idx]
                mb_ev = events_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]

                _, new_lp, entropy, new_val = model(mb_priv, mb_ev, action=mb_actions)

                # Ratio
                ratio = torch.exp(new_lp - mb_old_lp)

                # Surrogate losses
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_val, mb_returns)

                # Total loss
                total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                clip_losses.append(policy_loss.item())
                vf_losses.append(value_loss.item())
                ent_losses.append(entropy.mean().item())

        # Logging
        elapsed = time.time() - start_time
        fps = step_count / max(elapsed, 1e-4)
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        win_rate = (wins_count / max(episodes_completed, 1)) * 100.0

        print(
            f"Iter {iteration:3d}/{num_iterations} | "
            f"Steps: {step_count:6d} | "
            f"FPS: {fps:5.0f} | "
            f"Win Rate: {win_rate:4.1f}% | "
            f"Avg Reward: {avg_reward:+5.2f} | "
            f"Pol Loss: {np.mean(clip_losses):+6.3f} | "
            f"Val Loss: {np.mean(vf_losses):5.3f} | "
            f"Entropy: {np.mean(ent_losses):5.3f}"
        )

        # Periodic snapshot addition for curriculum
        if iteration % snapshot_interval == 0:
            snapshot_pool.add(model)
            ckpt_path = os.path.join(save_dir, f"benoit_iter_{iteration}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "iteration": iteration,
                    "encoder_type": encoder_type,
                },
                ckpt_path,
            )
            print(f" Saved snapshot to pool and checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, "benoit_final.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder_type": encoder_type,
        },
        final_path,
    )
    print(f"Training complete. Final model saved to {final_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Benoit ScrabbleBot with PPO self-play")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total training timesteps")
    parser.add_argument("--batch-size", type=int, default=2048, help="Rollout batch size")
    parser.add_argument("--encoder", type=str, default="transformer", choices=["transformer", "gru"])
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save models")
    args = parser.parse_args()

    train_benoit(
        total_timesteps=args.timesteps,
        steps_per_batch=args.batch_size,
        encoder_type=args.encoder,
        save_dir=args.save_dir,
    )
