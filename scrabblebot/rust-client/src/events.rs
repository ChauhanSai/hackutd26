//! Event ring buffer tracking public match history for Benoit.
//! Formats AuctionResult and WordPlay events into normalized float vectors.

use std::collections::VecDeque;

pub const EVENT_DIM: usize = 38;
pub const EVENT_HISTORY_LEN: usize = 32;

#[derive(Clone, Debug)]
pub struct EventBuffer {
    buffer: VecDeque<[f32; EVENT_DIM]>,
    capacity: usize,
}

impl Default for EventBuffer {
    fn default() -> Self {
        Self::new(EVENT_HISTORY_LEN)
    }
}

impl EventBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Record a public AuctionResult event:
    /// winner_idx: Some(0..5) or None if unsold
    /// letter: auctioned letter (e.g. 'A')
    /// paid: amount paid in currency
    pub fn record_auction_result(&mut self, winner_idx: Option<usize>, letter: char, paid: u64) {
        let mut ev = [0.0f32; EVENT_DIM];
        ev[1] = 1.0; // type = AUCTION_RESULT

        if let Some(w) = winner_idx {
            if w < 6 {
                ev[3 + w] = 1.0;
            } else {
                ev[3 + 6] = 1.0; // UNSOLD
            }
        } else {
            ev[3 + 6] = 1.0; // UNSOLD
        }

        let upper = letter.to_ascii_uppercase();
        let b = upper as u8;
        if (b'A'..=b'Z').contains(&b) {
            let ci = (b - b'A') as usize;
            ev[10 + ci] = 1.0;
        }

        ev[36] = (paid as f32) / 50.0;
        ev[37] = 0.0;

        self.push(ev);
    }

    /// Record a public WordPlay event:
    /// player_idx: player who played word (0..5)
    /// word: the word played
    /// total_reward: currency earned
    pub fn record_word_play(&mut self, player_idx: usize, word: &str, total_reward: u64) {
        let mut ev = [0.0f32; EVENT_DIM];
        ev[2] = 1.0; // type = WORD_PLAY

        if player_idx < 6 {
            ev[3 + player_idx] = 1.0;
        }

        for b in word.bytes() {
            let upper = b.to_ascii_uppercase();
            if (b'A'..=b'Z').contains(&upper) {
                let ci = (upper - b'A') as usize;
                ev[10 + ci] += 1.0 / 5.0;
            }
        }

        ev[36] = (total_reward as f32) / 50.0;
        ev[37] = (word.len() as f32) / 10.0;

        self.push(ev);
    }

    fn push(&mut self, event: [f32; EVENT_DIM]) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(event);
    }

    /// Outputs flat array [1, EVENT_HISTORY_LEN, EVENT_DIM] = [1216] floats
    /// Left-padded with zeros if fewer than capacity events recorded.
    pub fn to_history_array(&self) -> [f32; EVENT_HISTORY_LEN * EVENT_DIM] {
        let mut out = [0.0f32; EVENT_HISTORY_LEN * EVENT_DIM];
        let n = self.buffer.len();
        let offset = self.capacity.saturating_sub(n);

        for (i, ev) in self.buffer.iter().enumerate() {
            let start = (offset + i) * EVENT_DIM;
            out[start..start + EVENT_DIM].copy_from_slice(ev);
        }

        out
    }
}

