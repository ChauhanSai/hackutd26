# Benoit: Reinforcement Learning Scrabble-Bot (Rust Production Client)

"Benoit" is an autonomous, competitive AI bot for the Scrabble-style Vickrey auction game on [SpacetimeDB](https://spacetimedb.com).
The bot combines a deep Reinforcement Learning policy (trained via self-play PPO) with an exact programmatic anagram heuristic over the 172,820-word ENABLE dictionary.

---

## Architecture Overview

```
                                      ┌───────────────────────────────┐
                                      │   SpacetimeDB Game Server     │
                                      │  (Vickrey Auction & Reducers) │
                                      └──────────────┬────────────────┘
                                                     │ WebSocket (spacetimedb-sdk)
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Benoit Rust Client                               │
│                                                                             │
│  ┌─────────────────────────┐          ┌──────────────────────────────────┐  │
│  │   Public Event Buffer   │          │       Private Rack & State       │  │
│  │  (32-event ring buffer) │          │ (Balance, Letter Counts, Letter) │  │
│  └────────────┬────────────┘          └────────────────┬─────────────────┘  │
│               │                                        │                    │
│               └───────────────────┬────────────────────┘                    │
│                                   ▼                                         │
│                      ┌────────────────────────┐                             │
│                      │    tract-onnx Engine   │                             │
│                      │   (benoit.onnx model)  │                             │
│                      └────────────┬───────────┘                             │
│                                   ▼                                         │
│          ┌────────────────────────┴─────────────────────────┐               │
│          ▼                                                  ▼               │
│   Continuous Bid Amount                              Play Decision          │
│   (Scaled 0..1 to Balance)                   (Compare with Best Word Score) │
│          │                                                  │               │
│          ▼                                                  ▼               │
│   submit_bid(auction_id, bid)                 submit_word(match_id, word)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Exporting the Trained ONNX Model

Before launching the Rust client, train the model in the `training/` directory and export it to `benoit.onnx`:
See `scrabblebot/training/README.md`

---

## 2. Building the Rust Client

The client uses `tract-onnx`, a pure-Rust ONNX inference engine with zero external C++ library dependencies.

```bash
cd "scrabblebot/rust-client"

# Build release binary
cargo build --release
```

---

## 3. Running Against SpacetimeDB

### Running with a Team Bot Token
1. Generate your bot token at `https://scrabblebot.spacetimedb.com/team`.
2. Launch Benoit:

```bash
cd "scrabblebot/bot-starter"
BOT_NAME=alice BOT_TOKEN=<token> npm start
```

or, if `.token-<BOT_NAME>` exists

```bash
cd "scrabblebot/bot-starter"
BOT_NAME=<BOT_NAME> npm start
```

### Running with a Nonce (CLI)
```bash
BOT_NAME="benoit" \
BOT_NONCE="<NONCE_CODE>" \
cargo run --release
```

### Running verification tests
```bash
npm test --prefix "scrabblebot/bot-starter"
```

---

## 4. Key Files

- `src/main.rs`: Core event loop, state tracking, and reducer integration.
- `src/model.rs`: `tract-onnx` neural network execution plan and inference wrapper.
- `src/events.rs`: 32-event public history ring buffer encoding `AuctionResult` and `WordPlay`.
- `src/dictionary.rs`: Alphabet-ordered Trie anagram solver yielding sub-10-microsecond queries.

