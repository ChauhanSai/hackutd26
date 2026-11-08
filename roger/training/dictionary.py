"""
ENABLE dictionary and word scoring for Benoit ScrabbleBot.
Provides high-speed anagram search using an alphabet-ordered Trie.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple

# Standard Scrabble letter values
LETTER_VALUES: Dict[str, int] = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
    "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
    "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10,
}

# Standard 98-tile Scrabble distribution (no blanks)
DEFAULT_BAG: List[Tuple[str, int]] = [
    ("A", 9), ("B", 2), ("C", 2), ("D", 4), ("E", 12), ("F", 2), ("G", 3),
    ("H", 2), ("I", 9), ("J", 1), ("K", 1), ("L", 4), ("M", 2), ("N", 6),
    ("O", 8), ("P", 2), ("Q", 1), ("R", 6), ("S", 4), ("T", 6), ("U", 4),
    ("V", 2), ("W", 2), ("X", 1), ("Y", 2), ("Z", 1),
]

BAG_SIZE = 98


def length_multiplier(length: int) -> Tuple[int, int]:
    """
    Superlinear length multiplier represented as (numerator, denominator).
    Matches SpacetimeDB lengthMultiplier integer division:
      <= 3: 1.0x (1, 1)
      == 4: 1.5x (3, 2)
      == 5: 2.0x (2, 1)
      == 6: 2.5x (5, 2)
      >= 7: 3.0x (3, 1)
    """
    if length <= 3:
        return (1, 1)
    if length == 4:
        return (3, 2)
    if length == 5:
        return (2, 1)
    if length == 6:
        return (5, 2)
    return (3, 1)


def calc_word_score(word: str) -> Tuple[int, int, int]:
    """
    Calculate (base_score, bonus, total_reward) for a word.
    Uses integer division as in SpacetimeDB: total = (base * num) // denom.
    """
    base = sum(LETTER_VALUES.get(c, 0) for c in word.upper())
    num, denom = length_multiplier(len(word))
    total = (base * num) // denom
    bonus = total - base
    return base, bonus, total


class TrieNode:
    __slots__ = ("children", "word", "score", "base", "bonus")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.word: Optional[str] = None
        self.score: int = 0
        self.base: int = 0
        self.bonus: int = 0


class EnableDictionary:
    """
    Alphabet-ordered Trie index over the ENABLE wordlist (~173k words).
    By sorting letters of each word alphabetically, anagram searches skip permutations,
    yielding sub-50-microsecond queries per rack in pure Python.
    """

    def __init__(self, wordlist_path: Optional[str] = None) -> None:
        self.root = TrieNode()
        self.word_count = 0
        self.signatures_count = 0

        if wordlist_path is None:
            # Look relative to project root or current directory
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spacetimedb", "wordlist.txt")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scrabblebot", "spacetimedb", "wordlist.txt")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "wordlist.txt")),
                "/Users/saichauhan/Desktop/HackUTD '26/scrabblebot/spacetimedb/wordlist.txt",
                "/Users/saichauhan/Desktop/HackUTD '26/roger/spacetimedb/wordlist.txt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    wordlist_path = c
                    break

        if not wordlist_path or not os.path.exists(wordlist_path):
            raise FileNotFoundError(f"Could not locate wordlist.txt. Checked candidates: {candidates}")

        self._load_words(wordlist_path)

    def _load_words(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().upper()
                if len(w) < 2:
                    continue
                self.word_count += 1
                sig = "".join(sorted(w))
                curr = self.root
                for char in sig:
                    if char not in curr.children:
                        curr.children[char] = TrieNode()
                    curr = curr.children[char]

                base, bonus, total = calc_word_score(w)
                # Keep the highest scoring word for this anagram multiset
                if curr.word is None:
                    self.signatures_count += 1
                    curr.word = w
                    curr.score = total
                    curr.base = base
                    curr.bonus = bonus
                elif total > curr.score:
                    curr.word = w
                    curr.score = total
                    curr.base = base
                    curr.bonus = bonus

    def find_best_playable(
        self,
        rack: List[int] | Dict[str, int] | str,
        min_len: int = 2,
    ) -> Tuple[Optional[str], int, int, int]:
        """
        Find the highest-scoring valid word that can be formed from `rack`.
        `rack` can be:
          - A 26-element array/list of letter counts (A=0..Z=25)
          - A dict mapping char -> count
          - A string of uppercase letters e.g. "RETINAS"

        Returns: (best_word, best_score, best_len, best_bonus)
        If no word >= min_len can be formed, returns (None, 0, 0, 0).
        """
        if isinstance(rack, str):
            sorted_chars = sorted(rack.upper())
        elif isinstance(rack, dict):
            chars = []
            for c, count in rack.items():
                chars.extend([c.upper()] * count)
            sorted_chars = sorted(chars)
        else:
            # 26-element list
            chars = []
            for i, count in enumerate(rack):
                if count > 0:
                    c = chr(ord("A") + i)
                    chars.extend([c] * count)
            sorted_chars = chars  # already sorted since i goes 0..25

        if len(sorted_chars) < min_len:
            return None, 0, 0, 0

        best_word: Optional[str] = None
        best_score = -1
        best_base = 0
        best_bonus = 0
        best_len = 0

        # Branch-and-bound DFS on the sorted anagram trie
        def dfs(node: TrieNode, idx: int) -> None:
            nonlocal best_word, best_score, best_base, best_bonus, best_len

            if node.word is not None and len(node.word) >= min_len:
                if node.score > best_score:
                    best_score = node.score
                    best_word = node.word
                    best_base = node.base
                    best_bonus = node.bonus
                    best_len = len(node.word)
                elif node.score == best_score:
                    # Break score ties by longer word (higher superlinear bonus potential)
                    if len(node.word) > best_len:
                        best_score = node.score
                        best_word = node.word
                        best_base = node.base
                        best_bonus = node.bonus
                        best_len = len(node.word)

            prev_c: Optional[str] = None
            n_chars = len(sorted_chars)
            for i in range(idx, n_chars):
                c = sorted_chars[i]
                if c == prev_c:
                    continue
                prev_c = c
                if c in node.children:
                    dfs(node.children[c], i + 1)

        dfs(self.root, 0)

        if best_word is None:
            return None, 0, 0, 0
        return best_word, best_score, best_len, best_bonus

