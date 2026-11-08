"""
Gymnasium environment for ScrabbleBot auction matches.
Optimized for high-throughput headless vectorization and self-play.
"""

from __future__ import annotations
import collections
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dictionary import (
    DEFAULT_BAG,
    LETTER_VALUES,
    EnableDictionary,
    calc_word_score,
)

AUCTION_RESERVE = 1
STARTING_BALANCE = 100
STARTING_RACK_SIZE = 5
DEFAULT_NUM_PLAYERS = 6
EVENT_HISTORY_LEN = 32

# Event feature dimension:
# 3 (type: PAD, AUCTION, WORD) + 7 (player: 0..5, UNSOLD=6) + 26 (letter/letters) + 1 (amount) + 1 (word_len) = 38
EVENT_DIM = 38
PRIVATE_DIM = 57  # balance(1) + rack(26) + auction_letter(26) + score(1) + len(1) + bonus(1) + round_frac(1)


class BaseOpponent:
    """Base class for rule-based bots matching SpacetimeDB sim bots."""

    def __init__(self, name: str) -> None:
        self.name = name

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        """Returns (bid_amount, should_play_best_word)."""
        raise NotImplementedError


class CheapskateOpponent(BaseOpponent):
    def __init__(self) -> None:
        super().__init__("Cheapskate")

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        val = LETTER_VALUES.get(letter, 1)
        jitter = random.randint(-1, 3)
        bid = max(min(val + 3 + jitter, balance), 0)
        word, score, length, _ = best_word_info
        play = word is not None and length >= 3
        return bid, play


class ValueBidderOpponent(BaseOpponent):
    def __init__(self) -> None:
        super().__init__("ValueBidder")

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        val = LETTER_VALUES.get(letter, 1)
        # Matches live market: 8-15 bids on regular tiles, up to 20 on high tiles
        base_bid = 8 + val
        if balance < 30:
            base_bid = max(4, val + 2)
        jitter = random.randint(-2, 3)
        bid = max(min(base_bid + jitter, balance), 0)
        word, score, length, _ = best_word_info
        play = word is not None and length >= 3
        return bid, play


class AggressiveOpponent(BaseOpponent):
    def __init__(self) -> None:
        super().__init__("Aggressive")

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        val = LETTER_VALUES.get(letter, 1)
        is_vowel = letter in "AEIOU"
        # Bids 12-24 when rich, matching live aggressive bots
        premium = 6 if is_vowel else 4
        if letter in "JQZX":
            premium += 8
        base_bid = 10 + val + premium if balance >= 40 else max(5, val + 4)
        jitter = random.randint(-2, 3)
        bid = max(min(base_bid + jitter, balance), 0)
        word, score, length, _ = best_word_info
        play = word is not None and length >= 3
        return bid, play


class HoarderOpponent(BaseOpponent):
    """Hoards letters to maximize 2.5x - 3.0x superlinear multiplier."""

    def __init__(self) -> None:
        super().__init__("Hoarder")

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        val = LETTER_VALUES.get(letter, 1)
        is_vowel = letter in "AEIOU"
        premium = 5 if is_vowel else 3
        if letter in "JQZXS":
            premium += 7
        base_bid = 9 + val + premium if balance >= 40 else max(4, val + 3)
        bid = max(min(base_bid + random.randint(-2, 2), balance), 0)
        word, score, length, bonus = best_word_info
        # Plays if 5+ letters, bonus >= 6, or late game
        play = word is not None and (length >= 5 or bonus >= 6)
        return bid, play


class MarketCompetitiveOpponent(BaseOpponent):
    """Replicates live lobby opponents that bid 12-25 early and play 4-6 letter words."""

    def __init__(self) -> None:
        super().__init__("MarketCompetitive")

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        val = LETTER_VALUES.get(letter, 1)
        is_vowel = letter in "AEIOU"
        is_high = letter in "JQZXSK"
        
        if balance >= 60:
            target = 12 + val + (6 if is_high else (4 if is_vowel else 2))
        elif balance >= 30:
            target = 8 + val + (4 if is_high else 1)
        else:
            target = max(3, val + 1)
            
        jitter = random.randint(-2, 3)
        bid = max(min(target + jitter, balance), 0)
        word, score, length, bonus = best_word_info
        play = word is not None and (length >= 4 or bonus >= 4 or balance < 25)
        return bid, play


HERCULE_LETTER_AFFINITY: Dict[str, float] = {
    "A": 0.0, "B": 0.0586, "C": 0.1811, "D": 0.0221, "E": 0.0051,
    "F": 0.1338, "G": 0.0, "H": 0.2165, "I": 0.0, "J": 0.2694,
    "K": 0.2589, "L": 0.0, "M": 0.0623, "N": 0.0, "O": 0.0,
    "P": 0.1637, "Q": 0.41, "R": 0.0, "S": 0.0, "T": 0.0,
    "U": 0.0, "V": 0.0284, "W": 0.0724, "X": 0.3226, "Y": 0.2401,
    "Z": 0.4191,
}

HERCULE_WORD_PRIORS: List[Dict[str, Any]] = [
    {"word": "PERIQUE", "reward": 54, "games": 4, "bots": 4},
    {"word": "JOINERY", "reward": 51, "games": 4, "bots": 3},
    {"word": "YANQUIS", "reward": 57, "games": 3, "bots": 3},
    {"word": "FAVORED", "reward": 42, "games": 5, "bots": 3},
    {"word": "FOVEATED", "reward": 45, "games": 3, "bots": 3},
    {"word": "QUANTIC", "reward": 54, "games": 3, "bots": 2},
    {"word": "WHARFED", "reward": 51, "games": 3, "bots": 2},
    {"word": "EXORDIA", "reward": 45, "games": 3, "bots": 2},
    {"word": "JANITOR", "reward": 42, "games": 3, "bots": 2},
    {"word": "JIVER", "reward": 30, "games": 3, "bots": 3},
]

FREQUENCY: List[int] = [
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1
]

SCORING_SETTINGS_HERCULE: Dict[str, Any] = {
    "common": 12,
    "other": 9,
    "rare": 7,
    "fraction": 0.55,
    "reserve": 8,
    "gain": 0.9,
    "cashOut": 18,
    "holdGain": 3,
    "denial": 0.75,
    "dualPremium": 4,
}

HERCULE_PRECOMPUTED_PRIORS: List[Tuple[str, int, List[Tuple[int, int, str]]]] = []
for _p in HERCULE_WORD_PRIORS:
    _word = _p["word"]
    _reward = int(_p["reward"])
    _counts: Dict[int, int] = {}
    for _c in _word:
        _ci = ord(_c) - ord("A")
        _counts[_ci] = _counts.get(_ci, 0) + 1
    _items = [(_ci, _count, chr(ord("A") + _ci)) for _ci, _count in _counts.items()]
    HERCULE_PRECOMPUTED_PRIORS.append((_word, _reward, _items))


def find_public_hercule_goal(
    rack: List[int], remaining: int
) -> Optional[Tuple[str, int, int, List[str], float]]:
    """
    Evaluates public word priors for Hercule offensive planning.
    Returns: (word, reward, missing_count, missing_letters, expected_gain) or None
    """
    best_goal = None
    best_gain = -1.0
    for word, reward, letter_needs in HERCULE_PRECOMPUTED_PRIORS:
        missing_letters: List[str] = []
        likelihood = 1.0
        held = 0
        possible = True
        for ci, need_count, char in letter_needs:
            if need_count > FREQUENCY[ci]:
                possible = False
                break
            have_count = rack[ci]
            lack = max(0, need_count - have_count)
            held += min(need_count, have_count)
            for _ in range(lack):
                missing_letters.append(char)
            if lack > 0:
                likelihood *= min(1.0, FREQUENCY[ci] / 5.0) ** lack

        missing = len(missing_letters)
        if not possible or held < 4 or missing > 3 or missing > max(0, remaining):
            continue

        factor = 1.0 if missing == 0 else (0.7 if missing == 1 else (0.35 if missing == 2 else 0.12))
        expected_gain = reward * factor * likelihood * min(
            1.0, max(0.0, remaining - 1.0) / max(1.0, missing * 8.0)
        )
        if expected_gain > best_gain:
            best_gain = expected_gain
            best_goal = (word, reward, missing, missing_letters, expected_gain)
    return best_goal


class HerculeOpponent(BaseOpponent):
    """
    High-skill league baseline opponent wrapping the Hercule bot logic.
    Features:
      - Superlinear multiplier-driven aggressive word planning & cashing
      - Public-word priors (PERIQUE, JOINERY, YANQUIS, etc.)
      - High-tier letter affinity bids
      - Defensive bankroll reserve & liquidity capping
    """

    def __init__(
        self,
        name: str = "Hercule",
        dictionary: Optional[EnableDictionary] = None,
        max_extra: int = 6,
    ) -> None:
        super().__init__(name)
        self.dict = dictionary or EnableDictionary()
        self.max_extra = max_extra
        self.round_num = 0

    def reset(self) -> None:
        self.round_num = 0

    def act(
        self,
        player_idx: int,
        letter: str,
        balance: int,
        rack: List[int],
        best_word_info: Tuple[Optional[str], int, int, int],
    ) -> Tuple[int, bool]:
        self.round_num += 1
        letter = letter.upper()
        if not ("A" <= letter <= "Z") or balance < 1:
            return 0, False

        remaining = max(0, 68 - self.round_num)
        progress = min(1.0, self.round_num / 68.0)
        cli = ord(letter) - ord("A")
        have = rack[cli]
        size = sum(rack)
        ending = remaining <= 8 or progress >= 0.88

        # 1. Baseline own-value gain & goal potential
        score_before = best_word_info[1] if best_word_info[0] is not None else 0
        rack_plus = list(rack)
        rack_plus[cli] += 1
        _, score_after, _, _ = self.dict.find_best_playable(rack_plus, min_len=2)
        own_gain = max(0, score_after - score_before)

        goal_before = find_public_hercule_goal(rack, remaining)
        goal_after = find_public_hercule_goal(rack_plus, remaining)

        exp_before = goal_before[4] if goal_before is not None else 0.0
        exp_after = goal_after[4] if goal_after is not None else 0.0
        own_potential = max(0.0, exp_after - exp_before)

        # 2. Compute ownDesire
        common = letter in "AEIOUNRSTL"
        val = LETTER_VALUES.get(letter, 1)
        if common:
            explore = SCORING_SETTINGS_HERCULE["common"]  # 12
        elif val >= 8:
            explore = SCORING_SETTINGS_HERCULE["rare"]    # 7
        else:
            explore = SCORING_SETTINGS_HERCULE["other"]   # 9

        if have >= 2:
            explore = int(explore * 0.25)

        vowels = sum(rack[ord(c) - ord("A")] for c in "AEIOU")
        if size >= 4:
            if (vowels / size < 0.25 and letter in "AEIOU") or (vowels / size > 0.55 and letter not in "AEIOU"):
                explore += 2

        if size >= 9 and own_gain == 0 and own_potential < 2.0:
            explore = min(explore, 2)

        if ending and own_gain == 0:
            explore = 0

        gain_val = max(float(own_gain), own_potential)
        desire = max(
            explore,
            int(math.ceil(SCORING_SETTINGS_HERCULE["gain"] * gain_val + (2 if own_gain > 0 else 0)))
        )

        # Budget calculation
        reserve = 0 if ending else min(SCORING_SETTINGS_HERCULE["reserve"], int(balance * 0.15))
        budget_frac = 0.95 if ending else SCORING_SETTINGS_HERCULE["fraction"]
        budget = max(0, min(balance - reserve, max(1, int(balance * budget_frac))))
        baseline = min(budget, desire)

        # 3. Hercule public-word and affinity lifts
        affinity = HERCULE_LETTER_AFFINITY.get(letter, 0.0)
        productive = max(float(own_gain), own_potential) >= 8.0
        advances = (goal_before is not None) and (letter in goal_before[3]) and (goal_before[2] <= 2)

        goal_lift = max(0.0, exp_after - exp_before) * 0.75 if (advances and goal_before is not None) else 0.0
        letter_lift = (self.max_extra * affinity * 1.8) if productive else 0.0
        desired = min(self.max_extra, int(math.ceil(max(goal_lift, letter_lift))))

        if have >= 2 or (remaining <= 8 and own_gain == 0):
            desired = 0

        cap_frac = 0.95 if ending else 0.65
        cap = max(0, min(balance - reserve, int(balance * cap_frac)))
        bid_amount = max(baseline, min(cap, baseline + desired))
        bid = max(0, min(balance, bid_amount))

        # 4. Word play decision
        play = False
        word, score, length, bonus = best_word_info
        if word is not None:
            if length >= 7:
                play = True
            else:
                flush = (
                    balance <= SCORING_SETTINGS_HERCULE["cashOut"]
                    or remaining <= 5
                    or progress >= 0.92
                    or size >= 13
                )
                if flush:
                    play = True
                elif (
                    goal_before is not None
                    and goal_before[4] >= max(SCORING_SETTINGS_HERCULE["holdGain"], score * 0.12)
                    and remaining > 8
                    and progress < 0.86
                ):
                    # Hold tiles for the prior goal
                    play = False
                elif length >= 5 or (score >= 8 and (score / max(length, 1)) >= 3.5) or size >= 10:
                    play = True

        return bid, play


class ScrabbleBotEnv(gym.Env):
    """
    Gymnasium environment for a 6-player Scrabble-style Vickrey auction game.
    Agent 0 is the learning agent; agents 1..5 are opponents (simulated or self-play policies).
    Observation space provides private state and sequential public event history.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        dictionary: Optional[EnableDictionary] = None,
        num_players: int = DEFAULT_NUM_PLAYERS,
        history_len: int = EVENT_HISTORY_LEN,
        flatten_obs: bool = False,
        reward_type: str = "rank",  # "rank", "win_loss", "score_margin"
        opponents: Optional[List[BaseOpponent]] = None,
    ) -> None:
        super().__init__()
        self.dict = dictionary or EnableDictionary()
        self.num_players = num_players
        self.history_len = history_len
        self.flatten_obs = flatten_obs
        self.reward_type = reward_type

        # Action space: 2 continuous values in [0, 1]
        # action[0]: bid fraction (0.0 to 1.0 of current balance)
        # action[1]: play decision threshold (0.0 to 1.0, e.g. play if > 0.5 or bonus threshold)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation spaces
        self.private_space = spaces.Box(
            low=0.0,
            high=500.0,
            shape=(PRIVATE_DIM,),
            dtype=np.float32,
        )
        self.events_space = spaces.Box(
            low=0.0,
            high=500.0,
            shape=(self.history_len, EVENT_DIM),
            dtype=np.float32,
        )

        if self.flatten_obs:
            self.observation_space = spaces.Box(
                low=0.0,
                high=500.0,
                shape=(PRIVATE_DIM + self.history_len * EVENT_DIM,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict({
                "private": self.private_space,
                "events": self.events_space,
            })

        # Opponents setup
        self.opponents: List[BaseOpponent] = []
        if opponents is not None:
            self.set_opponents(opponents)
        else:
            self._init_default_opponents()

        # Match state
        self.bag: List[str] = []
        self.balances = np.zeros(self.num_players, dtype=np.int32)
        self.scores = np.zeros(self.num_players, dtype=np.int32)
        self.racks = [np.zeros(26, dtype=np.int32) for _ in range(self.num_players)]
        self.best_words: List[Tuple[Optional[str], int, int, int]] = [
            (None, 0, 0, 0) for _ in range(self.num_players)
        ]
        self.current_letter: Optional[str] = None
        self.round_num = 0
        self.event_buffer = collections.deque(maxlen=self.history_len)

    def _init_default_opponents(self) -> None:
        pool = [
            HerculeOpponent(dictionary=self.dict),
            MarketCompetitiveOpponent(),
            AggressiveOpponent(),
            ValueBidderOpponent(),
            HoarderOpponent(),
        ]
        self.opponents = pool[: self.num_players - 1]

    def set_opponents(self, opponents: List[BaseOpponent]) -> None:
        assert len(opponents) == self.num_players - 1
        self.opponents = opponents

    def _draw_letter(self) -> Optional[str]:
        if not self.bag:
            return None
        idx = random.randrange(len(self.bag))
        return self.bag.pop(idx)

    def _update_best_word(self, player_idx: int) -> None:
        if player_idx == 0:
            is_late_game = self.round_num >= 55 or len(self.bag) < 15
            min_len = 3 if is_late_game else 5
            best = self.dict.find_best_playable(self.racks[0], min_len=min_len)
            if best[0] is None and int(self.racks[0].sum()) >= 11 and not is_late_game:
                best = self.dict.find_best_playable(self.racks[0], min_len=4)
            self.best_words[0] = best
        else:
            self.best_words[player_idx] = self.dict.find_best_playable(
                self.racks[player_idx], min_len=2
            )

    def _record_event(self, event_vector: np.ndarray) -> None:
        self.event_buffer.append(event_vector)

    def _encode_auction_result(self, winner_idx: Optional[int], letter: str, paid: int) -> np.ndarray:
        ev = np.zeros(EVENT_DIM, dtype=np.float32)
        ev[1] = 1.0  # type = AUCTION_RESULT
        if winner_idx is not None and 0 <= winner_idx < self.num_players:
            ev[3 + winner_idx] = 1.0
        else:
            ev[3 + 6] = 1.0  # UNSOLD
        char_idx = ord(letter.upper()) - ord("A")
        if 0 <= char_idx < 26:
            ev[10 + char_idx] = 1.0
        ev[36] = paid / 50.0
        ev[37] = 0.0
        return ev

    def _encode_word_play(self, player_idx: int, word: str, total_reward: int) -> np.ndarray:
        ev = np.zeros(EVENT_DIM, dtype=np.float32)
        ev[2] = 1.0  # type = WORD_PLAY
        ev[3 + player_idx] = 1.0
        for c in word:
            ci = ord(c.upper()) - ord("A")
            if 0 <= ci < 26:
                ev[10 + ci] += 1.0 / 5.0
        ev[36] = total_reward / 50.0
        ev[37] = len(word) / 10.0
        return ev

    def _get_observation(self, player_idx: int = 0) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        # Private state:
        # balance(1) + rack(26) + current_letter(26) + score(1) + len(1) + bonus(1) + round_frac(1) = 57
        priv = np.zeros(PRIVATE_DIM, dtype=np.float32)
        priv[0] = self.balances[player_idx] / 100.0
        priv[1:27] = self.racks[player_idx] / 5.0
        if self.current_letter:
            cli = ord(self.current_letter) - ord("A")
            if 0 <= cli < 26:
                priv[27 + cli] = 1.0
        _, best_s, best_l, best_b = self.best_words[player_idx]
        priv[53] = best_s / 50.0
        priv[54] = best_l / 10.0
        priv[55] = best_b / 30.0
        priv[56] = min(self.round_num / 70.0, 1.0)

        # Public event history:
        events = np.zeros((self.history_len, EVENT_DIM), dtype=np.float32)
        buf_list = list(self.event_buffer)
        num_ev = len(buf_list)
        if num_ev > 0:
            events[self.history_len - num_ev :] = np.array(buf_list, dtype=np.float32)

        if self.flatten_obs:
            return np.concatenate([priv, events.flatten()])
        return {"private": priv, "events": events}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build standard 98-tile bag
        self.bag = []
        for char, count in DEFAULT_BAG:
            self.bag.extend([char] * count)
        random.shuffle(self.bag)

        self.balances = np.full(self.num_players, STARTING_BALANCE, dtype=np.int32)
        self.scores = np.zeros(self.num_players, dtype=np.int32)
        self.racks = [np.zeros(26, dtype=np.int32) for _ in range(self.num_players)]
        self.round_num = 0
        self.event_buffer.clear()

        # Deal starting rack of 5 tiles to each player (private)
        for p in range(self.num_players):
            for _ in range(STARTING_RACK_SIZE):
                tile = self._draw_letter()
                if tile is not None:
                    self.racks[p][ord(tile) - ord("A")] += 1
            self._update_best_word(p)

        # Draw the first auction letter
        self.current_letter = self._draw_letter()
        self.round_num = 1

        obs = self._get_observation(0)
        info = {
            "round": self.round_num,
            "bag_remaining": len(self.bag),
            "auction_letter": self.current_letter,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, Any]]:
        """
        Execute one Vickrey auction turn and word play decisions.
        `action` is for Agent 0: [bid_fraction in [0, 1], play_threshold in [0, 1]].
        """
        assert self.current_letter is not None, "Cannot step on terminated environment"

        # 1. Collect bids from all players
        bids = np.zeros(self.num_players, dtype=np.int32)

        # Agent 0 action
        # Clamp bid fraction to at most 25% of balance (max 25 coins)
        # to prevent Vickrey auction price-gouging and economic suicide
        bid_frac = float(np.clip(action[0], 0.0, 1.0)) * 0.25
        agent0_bid = min(25, int(round(bid_frac * self.balances[0])))
        bids[0] = agent0_bid
        agent0_play = float(action[1])

        # Opponent actions (Agents 1..5)
        opponent_plays: List[bool] = []
        for i, opp in enumerate(self.opponents):
            p_idx = i + 1
            b, p = opp.act(
                player_idx=p_idx,
                letter=self.current_letter,
                balance=int(self.balances[p_idx]),
                rack=self.racks[p_idx].tolist(),
                best_word_info=self.best_words[p_idx],
            )
            bids[p_idx] = b
            opponent_plays.append(p)

        # 2. Resolve Vickrey Auction
        # Non-zero bidders sorted descending with random tie breaking
        valid_bidders = [(p, bids[p]) for p in range(self.num_players) if bids[p] > 0]
        random.shuffle(valid_bidders)  # shuffle for fair tie-breaking
        valid_bidders.sort(key=lambda x: x[1], reverse=True)

        winner_idx: Optional[int] = None
        paid = 0
        if valid_bidders:
            winner_idx = valid_bidders[0][0]
            if len(valid_bidders) > 1:
                # Vickrey payment: 2nd highest bid
                paid = valid_bidders[1][1]
            else:
                # Single bidder pays reserve
                paid = min(AUCTION_RESERVE, valid_bidders[0][1])

            # Deduct payment and add tile to rack
            if self.balances[winner_idx] >= paid:
                self.balances[winner_idx] -= paid
                char_idx = ord(self.current_letter) - ord("A")
                self.racks[winner_idx][char_idx] += 1
                self._update_best_word(winner_idx)
            else:
                winner_idx = None
                paid = 0
        else:
            # Tile unsold: return to bag as per ScrabbleBot rules
            self.bag.append(self.current_letter)

        # Record public AuctionResult event
        ev_auction = self._encode_auction_result(winner_idx, self.current_letter, paid)
        self._record_event(ev_auction)

        # 3. Word Play Resolution
        # Agent 0 word play decision:
        word0, score0, len0, bonus0 = self.best_words[0]
        if word0 is not None:
            is_late_game = self.round_num >= 55 or len(self.bag) < 15
            # Enforce hardcoded minimum word length safety net:
            # Must be len0 >= 6, OR (len0 >= 5 and (bonus0 >= 4 or agent0_play > 0.45)), OR (is_late_game and len0 >= 3)
            # This completely avoids playing 1.0x-1.5x words during early/mid game.
            should_play0 = (
                len0 >= 6 or
                (len0 >= 5 and (bonus0 >= 4 or agent0_play > 0.45)) or
                (is_late_game and len0 >= 3) or
                (int(self.racks[0].sum()) >= 11 and len0 >= 4)
            )
            if should_play0:
                self._play_word(0, word0, score0)

        # Opponent word plays
        for i, should_play in enumerate(opponent_plays):
            p_idx = i + 1
            w, s, l, _ = self.best_words[p_idx]
            if w is not None:
                is_late_opp = self.round_num > 45 or len(self.bag) < 20
                if should_play or (is_late_opp and l >= 3) or (self.balances[p_idx] < 20 and l >= 3):
                    self._play_word(p_idx, w, s)

        # 4. Check Match Termination
        terminated = False
        reward = 0.0

        next_letter = self._draw_letter()
        if next_letter is None:
            # Bag is empty -> Game Over!
            terminated = True
            self.current_letter = None
            reward = self._calculate_terminal_reward(0)
        else:
            self.current_letter = next_letter
            self.round_num += 1

        obs = self._get_observation(0)
        info = {
            "round": self.round_num,
            "balances": self.balances.copy(),
            "scores": self.scores.copy(),
            "bag_remaining": len(self.bag),
            "winner_idx": winner_idx,
            "paid": paid,
        }
        if terminated:
            ranks = self._get_final_ranks()
            info["agent0_rank"] = ranks[0]
            info["ranks"] = ranks

        return obs, reward, terminated, False, info

    def _play_word(self, player_idx: int, word: str, total_reward: int) -> None:
        # Deduct letters from rack
        for c in word:
            ci = ord(c.upper()) - ord("A")
            self.racks[player_idx][ci] -= 1
        # Update balance and score
        self.balances[player_idx] += total_reward
        self.scores[player_idx] += total_reward
        self._update_best_word(player_idx)

        # Record public WordPlay event
        ev_word = self._encode_word_play(player_idx, word, total_reward)
        self._record_event(ev_word)

    def _get_final_ranks(self) -> List[int]:
        """Compute final placement ranks (1 to 6) based on score, breaking ties by balance."""
        player_scores = [
            (p, int(self.scores[p]), int(self.balances[p]))
            for p in range(self.num_players)
        ]
        # Sort descending by score, then balance
        player_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ranks = [0] * self.num_players
        for rank_pos, (p, _, _) in enumerate(player_scores):
            ranks[p] = rank_pos + 1
        return ranks

    def _calculate_terminal_reward(self, player_idx: int = 0) -> float:
        """
        Terminal reward optimizing for 6-player win rate and relative placement.
        Sparse: 0 on all non-terminal steps.
        Enforces severe penalty for 0 score and penalizes holding excessive unplayed tiles at match end.
        """
        ranks = self._get_final_ranks()
        rank = ranks[player_idx]
        my_score = int(self.scores[player_idx])
        unplayed_tiles = int(self.racks[player_idx].sum())

        # Crushing penalty for scoring 0 points
        if my_score == 0:
            return -2.0

        if self.reward_type == "win_loss":
            return 1.0 if rank == 1 else -1.0

        if self.reward_type == "score_margin":
            opp_scores = [float(self.scores[i]) for i in range(self.num_players) if i != player_idx]
            mean_opp = sum(opp_scores) / len(opp_scores)
            return float(np.clip((my_score - mean_opp) / 50.0, -2.0, 2.0))

        # Default "rank": Smooth rank-based win-rate optimization in [-1.0, 1.0]
        # with score bonus to encourage aggressive point production
        rank_rewards = {1: 1.0, 2: 0.5, 3: 0.2, 4: -0.1, 5: -0.4, 6: -0.8}
        base_reward = rank_rewards.get(rank, -0.8)
        score_bonus = min(0.6, my_score * 0.006)
        
        # Penalty for leaving tiles unplayed on rack at game end (waste of capital)
        tile_waste_penalty = max(0.0, (unplayed_tiles - 2) * 0.04)

        return float(base_reward + score_bonus - tile_waste_penalty)

