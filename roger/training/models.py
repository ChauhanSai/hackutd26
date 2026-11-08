"""
Neural network architectures for Benoit ScrabbleBot.
Supports both lightweight Transformer and Recurrent (GRU/LSTM) encoders
for implicit opponent and bag tracking from public event history streams.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from environment import EVENT_DIM, EVENT_HISTORY_LEN, PRIVATE_DIM


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence events."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EventTransformerEncoder(nn.Module):
    """
    Lightweight Transformer Encoder to process sequential public event history.
    Learns implicit tracking of opponent racks, balances, and bag distribution.
    """

    def __init__(
        self,
        event_dim: int = EVENT_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(event_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=EVENT_HISTORY_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: [batch, seq_len, event_dim]
        h = self.input_proj(events)
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = self.norm(h)
        # Sequence pooling: mean pool over non-padding steps
        # or global average pool over sequence length
        return torch.mean(h, dim=1)  # [batch, d_model]


class EventRNNEncoder(nn.Module):
    """
    Lightweight GRU Encoder for event sequence processing.
    """

    def __init__(
        self,
        event_dim: int = EVENT_DIM,
        d_model: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(event_dim, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: [batch, seq_len, event_dim]
        h = F.relu(self.input_proj(events))
        out, _ = self.gru(h)
        last_step = out[:, -1, :]  # [batch, d_model]
        return self.norm(last_step)


class BenoitActorCritic(nn.Module):
    """
    Unified Actor-Critic architecture for Benoit ScrabbleBot.
    Combines:
      - Private State Encoder (balance, rack, auction tile, best word)
      - Public Event History Encoder (Transformer or GRU)
      - Continuous Actor Head: [BidAmount in [0, 1], PlayWordThreshold in [0, 1]]
      - Critic Head: State Value estimate V(s) for PPO Generalized Advantage Estimation
    """

    def __init__(
        self,
        private_dim: int = PRIVATE_DIM,
        event_dim: int = EVENT_DIM,
        history_len: int = EVENT_HISTORY_LEN,
        encoder_type: str = "transformer",  # "transformer" or "gru"
        d_model: int = 64,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.private_dim = private_dim
        self.event_dim = event_dim
        self.history_len = history_len
        self.encoder_type = encoder_type

        # 1. Public Event Encoder
        if encoder_type == "transformer":
            self.event_encoder = EventTransformerEncoder(
                event_dim=event_dim,
                d_model=d_model,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
            )
        elif encoder_type == "gru":
            self.event_encoder = EventRNNEncoder(
                event_dim=event_dim,
                d_model=d_model,
                num_layers=1,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 2. Private State Encoder
        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # 3. Fusion Trunk
        fusion_dim = d_model * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
        )

        # 4. Actor (Policy) Head
        # Outputs mean for Gaussian action distribution (squashed to [0, 1] via Sigmoid)
        self.actor_mean = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        # Learnable log standard deviation for exploration
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # 5. Critic (Value) Head
        self.critic = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def _parse_inputs(
        self,
        private_or_obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        events: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supports Dict observation, separate tensors, or flattened tensor."""
        if isinstance(private_or_obs, dict):
            priv = private_or_obs["private"]
            ev = private_or_obs["events"]
        elif events is not None:
            priv = private_or_obs
            ev = events
        else:
            # Flattened single tensor: [batch, private_dim + history_len * event_dim]
            flat = private_or_obs
            priv = flat[:, : self.private_dim]
            ev_flat = flat[:, self.private_dim :]
            ev = ev_flat.view(-1, self.history_len, self.event_dim)
        return priv, ev

    def _get_features(self, priv: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
        ev_feat = self.event_encoder(ev)
        priv_feat = self.private_encoder(priv)
        fused = torch.cat([priv_feat, ev_feat], dim=-1)
        return self.fusion(fused)

    def forward(
        self,
        private: torch.Tensor,
        events: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for PPO training.
        Returns: (actions, log_prob, entropy, state_value)
        """
        features = self._get_features(private, events)
        mean_raw = self.actor_mean(features)
        std = torch.exp(self.actor_log_std).expand_as(mean_raw)

        dist = Normal(mean_raw, std)

        if action is None:
            # Sample in unconstrained space, then squash to [0, 1]
            raw_action = dist.rsample()
            action_clamped = torch.sigmoid(raw_action)
        else:
            # Inverse sigmoid (logit) for log_prob evaluation
            # Clip for numerical stability
            action_bounded = torch.clamp(action, 1e-6, 1.0 - 1e-6)
            raw_action = torch.log(action_bounded / (1.0 - action_bounded))
            action_clamped = action

        # Compute log probability with change-of-variables jacobian correction:
        # log p(y) = log p(x) - sum(log(y * (1 - y))) where y = sigmoid(x)
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        sigmoid_deriv = torch.log(action_clamped * (1.0 - action_clamped) + 1e-8).sum(dim=-1)
        log_prob = log_prob_raw - sigmoid_deriv

        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action_clamped, log_prob, entropy, value

    def get_value(self, private: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        features = self._get_features(private, events)
        return self.critic(features).squeeze(-1)

    def act_deterministic(
        self,
        private: torch.Tensor,
        events: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic inference mode for ONNX export and tournament play.
        Returns: (action_clamped [batch, 2], value [batch, 1])
        """
        features = self._get_features(private, events)
        mean_raw = self.actor_mean(features)
        action_clamped = torch.sigmoid(mean_raw)
        value = self.critic(features)
        return action_clamped, value

