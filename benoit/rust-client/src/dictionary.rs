//! ENABLE Dictionary and word anagram solver for Benoit in Rust.
//! Uses a sorted-character Trie for microsecond word lookup.

use std::collections::HashMap;

pub const LETTER_VALUES: [u32; 26] = [
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10,
];

#[inline]
pub fn letter_value(c: char) -> u32 {
    let u = c as u32;
    if (65..=90).contains(&u) {
        LETTER_VALUES[(u - 65) as usize]
    } else {
        0
    }
}

/// Superlinear length multiplier as (numerator, denominator)
/// Matches SpacetimeDB lengthMultiplier integer division:
///   <= 3: 1.0x (1, 1)
///   == 4: 1.5x (3, 2)
///   == 5: 2.0x (2, 1)
///   == 6: 2.5x (5, 2)
///   >= 7: 3.0x (3, 1)
#[inline]
pub fn length_multiplier(len: usize) -> (u64, u64) {
    match len {
        0..=3 => (1, 1),
        4 => (3, 2),
        5 => (2, 1),
        6 => (5, 2),
        _ => (3, 1),
    }
}

pub fn calc_word_score(word: &str) -> (u64, u64, u64) {
    let mut base = 0u64;
    for b in word.bytes() {
        if b.is_ascii_uppercase() {
            base += LETTER_VALUES[(b - b'A') as usize] as u64;
        }
    }
    let (num, den) = length_multiplier(word.len());
    let total = (base * num) / den;
    let bonus = total.saturating_sub(base);
    (base, bonus, total)
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, TrieNode>,
    word: Option<String>,
    score: u64,
    base: u64,
    bonus: u64,
}

pub struct EnableDictionary {
    root: TrieNode,
    pub word_count: usize,
}

impl EnableDictionary {
    pub fn new_from_str(wordlist: &str) -> Self {
        let mut root = TrieNode::default();
        let mut count = 0;

        for line in wordlist.lines() {
            let w = line.trim();
            if w.len() < 2 {
                continue;
            }
            let upper = w.to_ascii_uppercase();
            count += 1;

            let mut sorted_bytes = upper.as_bytes().to_vec();
            sorted_bytes.sort_unstable();

            let mut curr = &mut root;
            for &b in &sorted_bytes {
                curr = curr.children.entry(b).or_default();
            }

            let (base, bonus, total) = calc_word_score(&upper);
            if curr.word.is_none() || total > curr.score {
                curr.word = Some(upper);
                curr.score = total;
                curr.base = base;
                curr.bonus = bonus;
            }
        }

        Self { root, word_count: count }
    }

    /// Finds the highest-scoring valid word playable from `rack` (26 letter counts).
    /// Returns: Option<(word, total_score, length, bonus)>
    pub fn find_best_playable(&self, rack: &[u8; 26], min_len: usize) -> Option<(String, u64, usize, u64)> {
        let mut chars = Vec::with_capacity(16);
        for (i, &count) in rack.iter().enumerate() {
            for _ in 0..count {
                chars.push(b'A' + i as u8);
            }
        }
        if chars.len() < min_len {
            return None;
        }

        let mut best_word: Option<String> = None;
        let mut best_score: u64 = 0;
        let mut best_base: u64 = 0;
        let mut best_bonus: u64 = 0;
        let mut best_len: usize = 0;

        fn dfs(
            node: &TrieNode,
            chars: &[u8],
            idx: usize,
            min_len: usize,
            best_word: &mut Option<String>,
            best_score: &mut u64,
            best_base: &mut u64,
            best_bonus: &mut u64,
            best_len: &mut usize,
        ) {
            if let Some(ref w) = node.word {
                if w.len() >= min_len {
                    if node.score > *best_score || (node.score == *best_score && w.len() > *best_len) {
                        *best_score = node.score;
                        *best_word = Some(w.clone());
                        *best_base = node.base;
                        *best_bonus = node.bonus;
                        *best_len = w.len();
                    }
                }
            }

            let mut prev_b: Option<u8> = None;
            for i in idx..chars.len() {
                let b = chars[i];
                if Some(b) == prev_b {
                    continue;
                }
                prev_b = Some(b);

                if let Some(child) = node.children.get(&b) {
                    dfs(child, chars, i + 1, min_len, best_word, best_score, best_base, best_bonus, best_len);
                }
            }
        }

        dfs(
            &self.root,
            &chars,
            0,
            min_len,
            &mut best_word,
            &mut best_score,
            &mut best_base,
            &mut best_bonus,
            &mut best_len,
        );

        best_word.map(|w| (w, best_score, best_len, best_bonus))
    }
}

