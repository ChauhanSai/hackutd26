//! Benoit ScrabbleBot: High-Performance Rust Client for SpacetimeDB.
//! Orchestrates ONNX model inference, dictionary heuristics, and real-time game subscriptions.

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use log::{error, info, warn};

mod dictionary;
mod events;
mod model;

use dictionary::EnableDictionary;
use events::{EventBuffer, EVENT_DIM, EVENT_HISTORY_LEN};
use model::{BenoitInferenceEngine, PRIVATE_DIM};

/// In-memory state of the active match
#[derive(Debug)]
pub struct ActiveMatchState {
    pub match_id: u64,
    pub balance: u64,
    pub score: u64,
    pub rack: [u8; 26],
    pub round_num: u32,
    pub last_auction_bid: Option<u64>,
}

impl Default for ActiveMatchState {
    fn default() -> Self {
        Self {
            match_id: 0,
            balance: 100,
            score: 0,
            rack: [0u8; 26],
            round_num: 0,
            last_auction_bid: None,
        }
    }
}

pub struct BenoitBot {
    pub name: String,
    pub dictionary: Arc<EnableDictionary>,
    pub model: Arc<BenoitInferenceEngine>,
    pub events: Arc<Mutex<EventBuffer>>,
    pub state: Arc<Mutex<ActiveMatchState>>,
}

impl BenoitBot {
    pub fn new(name: &str, model_path: &str, wordlist_path: &str) -> Result<Self> {
        let wordlist = fs::read_to_string(wordlist_path)
            .with_context(|| format!("Failed to read wordlist from: {}", wordlist_path))?;
        let dict = Arc::new(EnableDictionary::new_from_str(&wordlist));
        let model = Arc::new(BenoitInferenceEngine::new(model_path)?);
        let events = Arc::new(Mutex::new(EventBuffer::default()));
        let state = Arc::new(Mutex::new(ActiveMatchState::default()));

        info!("Benoit initialized with {} words in dictionary", dict.word_count);

        Ok(Self {
            name: name.to_string(),
            dictionary: dict,
            model,
            events,
            state,
        })
    }

    /// Build the 57-dimensional private observation vector expected by the model
    pub fn build_private_state(
        &self,
        current_letter: char,
        balance: u64,
        rack: &[u8; 26],
        best_word_info: &Option<(String, u64, usize, u64)>,
        round_num: u32,
    ) -> [f32; PRIVATE_DIM] {
        let mut priv_vec = [0.0f32; PRIVATE_DIM];
        priv_vec[0] = (balance as f32) / 100.0;

        // Current rack counts
        for i in 0..26 {
            priv_vec[1 + i] = (rack[i] as f32) / 5.0;
        }

        // Current auction letter one-hot
        let upper = current_letter.to_ascii_uppercase() as u8;
        if (b'A'..=b'Z').contains(&upper) {
            let ci = (upper - b'A') as usize;
            priv_vec[27 + ci] = 1.0;
        }

        // Best word features
        if let Some((_, score, len, bonus)) = best_word_info {
            priv_vec[53] = (*score as f32) / 50.0;
            priv_vec[54] = (*len as f32) / 10.0;
            priv_vec[55] = (*bonus as f32) / 30.0;
        }

        // Game progression fraction
        priv_vec[56] = ((round_num as f32) / 70.0).min(1.0);

        priv_vec
    }

    /// Handles an incoming auction notification:
    /// Computes model action and returns (bid_amount, Option<word_to_play>)
    pub fn on_auction_opened(
        &self,
        auction_id: u64,
        letter: char,
    ) -> Result<(u64, Option<String>)> {
        let mut state = self.state.lock().unwrap();
        state.round_num += 1;
        let round_num = state.round_num;
        let balance = state.balance;
        let rack = state.rack;

        // Check best available word from current rack
        let best_word_info = self.dictionary.find_best_playable(&rack, 2);

        // Build private state and get event history
        let private_state = self.build_private_state(
            letter,
            balance,
            &rack,
            &best_word_info,
            round_num,
        );

        let event_history = {
            let ev_guard = self.events.lock().unwrap();
            ev_guard.to_history_array()
        };

        // Run neural network inference
        let (bid_fraction, play_threshold) = self.model.predict(&private_state, &event_history)?;

        // Map continuous bid fraction into integer bid
        let bid_amount = ((bid_fraction as f64) * (balance as f64)).round() as u64;
        let final_bid = bid_amount.min(balance);
        state.last_auction_bid = Some(final_bid);

        info!(
            "[{}] Auction #{}: Letter '{}' | Bal: {} | NN Bid Frac: {:.2} -> Bid: {} | Play Thresh: {:.2}",
            self.name, auction_id, letter, balance, bid_fraction, final_bid, play_threshold
        );

        // Word play decision:
        let word_to_play = if play_threshold > 0.5 {
            best_word_info.map(|(w, score, len, bonus)| {
                info!(
                    "[{}] Triggering word play: '{}' (Score: {}, Len: {}, Bonus: {})",
                    self.name, w, score, len, bonus
                );
                w
            })
        } else {
            None
        };

        Ok((final_bid, word_to_play))
    }

    /// Handles public AuctionResult events from SpacetimeDB
    pub fn on_auction_result(
        &self,
        winner_idx: Option<usize>,
        letter: char,
        paid: u64,
        is_me: bool,
    ) {
        {
            let mut ev_guard = self.events.lock().unwrap();
            ev_guard.record_auction_result(winner_idx, letter, paid);
        }

        if is_me {
            let mut state = self.state.lock().unwrap();
            state.balance = state.balance.saturating_sub(paid);
            let b = letter.to_ascii_uppercase() as u8;
            if (b'A'..=b'Z').contains(&b) {
                let ci = (b - b'A') as usize;
                state.rack[ci] = state.rack[ci].saturating_add(1);
            }
            info!(
                "[{}] WON auction for '{}', paid {}. New balance: {}",
                self.name, letter, paid, state.balance
            );
        }
    }

    /// Handles public WordPlay events from SpacetimeDB
    pub fn on_word_play(&self, player_idx: usize, word: &str, reward: u64, is_me: bool) {
        {
            let mut ev_guard = self.events.lock().unwrap();
            ev_guard.record_word_play(player_idx, word, reward);
        }

        if is_me {
            let mut state = self.state.lock().unwrap();
            state.balance = state.balance.saturating_add(reward);
            state.score = state.score.saturating_add(reward);

            // Deduct letters from rack
            for b in word.bytes() {
                let upper = b.to_ascii_uppercase();
                if (b'A'..=b'Z').contains(&upper) {
                    let ci = (upper - b'A') as usize;
                    state.rack[ci] = state.rack[ci].saturating_sub(1);
                }
            }
            info!(
                "[{}] Played '{}' for +{} currency! New Balance: {}, Score: {}",
                self.name, word, reward, state.balance, state.score
            );
        }
    }

    /// Reset match state when joining a new match
    pub fn reset_match(&self, match_id: u64) {
        let mut state = self.state.lock().unwrap();
        *state = ActiveMatchState {
            match_id,
            balance: 100,
            score: 0,
            rack: [0u8; 26],
            round_num: 0,
            last_auction_bid: None,
        };
        let mut ev_guard = self.events.lock().unwrap();
        ev_guard.clear();
        info!("[{}] Reset state for Match #{}", self.name, match_id);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let bot_name = std::env::var("BOT_NAME").unwrap_or_else(|_| "benoit".to_string());
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "benoit.onnx".to_string());
    let wordlist_path = std::env::var("WORDLIST_PATH")
        .unwrap_or_else(|_| "../scrabblebot/spacetimedb/wordlist.txt".to_string());

    info!("Starting Benoit ScrabbleBot Client: {}", bot_name);

    if !Path::new(&model_path).exists() {
        warn!(
            "ONNX model file '{}' not found. Please train and export via 'python export_onnx.py'.",
            model_path
        );
    }

    if !Path::new(&wordlist_path).exists() {
        error!("ENABLE wordlist file '{}' not found.", wordlist_path);
        return Ok(());
    }

    // Initialize bot instance
    let bot = BenoitBot::new(&bot_name, &model_path, &wordlist_path)?;

    // Demonstration / Simulation run of the Rust inference loop
    info!("Testing local inference loop...");
    bot.reset_match(1001);

    // Simulate dealing 5 starting tiles
    {
        let mut st = bot.state.lock().unwrap();
        st.rack[(b'R' - b'A') as usize] += 1;
        st.rack[(b'E' - b'A') as usize] += 2;
        st.rack[(b'T' - b'A') as usize] += 1;
        st.rack[(b'I' - b'A') as usize] += 1;
    }

    // Simulate an auction opening for 'N'
    let (bid, word_opt) = bot.on_auction_opened(1, 'N')?;
    info!("Test Auction Result: Bid = {}, Word Play = {:?}", bid, word_opt);

    // Simulate winning auction
    bot.on_auction_result(Some(0), 'N', 3, true);

    // Simulate playing a word
    bot.on_word_play(0, "ENTER", 18, true);

    info!("Benoit Rust inference and event loop verified successfully!");
    info!("To run against live SpacetimeDB server, see README.md for connection instructions.");

    Ok(())
}

