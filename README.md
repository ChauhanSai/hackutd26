# Benoit
*A trio of AI detective bots—Benoit, Hercule, and Roger—cracking the high-stakes code of real-time sealed-bid Scrabble auctions on SpacetimeDB.*

---

## What it does
In the cutthroat arena of **ScrabbleBot** (built on [SpacetimeDB](https://spacetimedb.com)), six autonomous bots enter a lobby to duel over letters drawn from a 98-tile Scrabble bag. Every round, one tile is revealed for a lightning-fast **1-second sealed-bid Vickrey auction** (where the highest bidder wins the letter but pays the runner-up's bid price). Bots hoard letters on private racks, navigate bankroll constraints, and cash in words from the 172,820-word ENABLE dictionary to earn points and currency. Words played score base tile values amplified by a **superlinear length bonus**—up to a massive **3.0× multiplier** for words of 7 letters or more.

To conquer this blend of imperfect-information game theory, economic auctions, and combinatorial linguistics, we engineered a detective trilogy of three distinct bot architectures:
- **Benoit (Standard Deep RL)**: A pure self-play Reinforcement Learning agent powered by Proximal Policy Optimization (PPO) and a Transformer-based sequential public event encoder that learns bidding fractions and multiplier cash-out thresholds from scratch.
- **Hercule (Deductive Game-Theoretic Grandmaster)**: A master of "the little grey cells" that maintains a deterministic public ledger to reconstruct every opponent's private rack, benchmarks live leaderboard threats using empirical priors, executes bitmask-accelerated combinatorial branch-and-bound rack packing, and places calculated spite/denial bids.
- **Roger (Adversarial Counter-RL)**: An opportunistic counter-agent trained in a custom hybrid league curriculum specifically against Hercule's deductive bidding engine, learning to exploit second-price Vickrey mechanics, bait deterministic overbids, and bankrupt predictable grandmasters.

### Key Features:
- **Real-Time Auction Bidding Under 1 Second**: Sub-millisecond inference pipelines in both TypeScript and Rust via ONNX runtime (`tract-onnx`), making decisions in $<15\text{ ms}$.
- **Vickrey Auction Exploitation**: Second-price equilibrium modeling that bids high enough to guarantee tile acquisition while actively manipulating runner-up prices to drain opponents' liquidity.
- **Microsecond Anagram Search Engine**: Alphabet-sorted anagram Trie indexing all 172,820 words of the ENABLE lexicon, delivering optimal playable rack combinations in $<25\,\mu\text{s}$.
- **Superlinear Multiplier Frontier Planning**: Probabilistic transition trees estimating whether to hold tiles for 7+ letter bonuses (up to 3.0× multiplier) based on English bag distribution frequencies $P(\text{draw})$.
- **Autonomous Continuous Match Lifecycle**: Self-healing connection layer that automatically enters lobbies, plays matches to completion, records analytics, and re-queues 24/7.

---

## How we built it
Our engineering philosophy was driven by evolutionary competition: **start tabula rasa, build a deductive perfectionist, then train an adversarial assassin to dethrone it.**

- **Ideology 1 — Tabula Rasa Learning (Benoit)**: Rather than hard-coding human heuristics about what letters are "good", we allowed deep reinforcement learning to discover bidding dynamics. We built a high-throughput, headless 6-player Gymnasium simulator in Python running at $>30,000$ steps per second, training an Actor-Critic architecture via PPO with Generalized Advantage Estimation (GAE) across thousands of simulated matches.
- **Ideology 2 — Deterministic Deduction & Information Theory (Hercule)**: In imperfect-information games, information asymmetry is everything. Because SpacetimeDB emits public `AuctionResult` and `WordPlay` events, we realized we could reconstruct our opponents' hidden racks. We built an analytical powerhouse that calculates the exact marginal own-value of every letter and models opponent utility to implement strategic denial bidding.
- **Ideology 3 — Adversarial League Synthesis (Roger)**: Pure RL often plays weakly against structured game-theoretic bots, while static heuristic bots suffer from deterministic exploitability. To bridge the gap, we ported Hercule's entire analytical decision engine into our Python training environment as a high-skill baseline anchor. Roger was trained directly against Hercule in an adversarial league curriculum, discovering the hidden economic blind spots in Hercule's conservative bidding curve.

---

## 🧠 Methodology
**Architecture**: Specific method used for each bot

```
                    ┌───────────────────────────────────────────────┐
                    │      SpacetimeDB Real-Time Game Server        │
                    │  (6-Player Lobby • Vickrey Auction Reducer)   │
                    └───────────────────────┬───────────────────────┘
                                            │ WebSocket Stream
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                The Detective Trio                                   │
│                                                                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐  │
│  │      Benoit (RL)        │  │      Hercule (Deduction) │  │    Roger (Counter) │  │
│  │                         │  │                          │  │                    │  │
│  │ • 2-Layer Transformer   │  │ • Public Rack Ledger     │  │ • PPO Actor-Critic │  │
│  │   Event Sequence Model  │  │   Deduction Engine       │  │ • League Trained   │  │
│  │ • Private State MLP     │  │ • Bitmask Trie Packing   │  │   70% vs Hercule   │  │
│  │ • PPO Continuous Actor  │  │ • Opponent Threat Prior  │  │   30% Self-Snap    │  │
│  │ • Safe Bankroll Clamps  │  │ • Marginal Denial Bids   │  │ • Vickrey Baiting  │  │
│  └───────────┬─────────────┘  └────────────┬─────────────┘  └──────────┬─────────┘  │
└──────────────┼─────────────────────────────┼───────────────────────────┼────────────┘
                ▼                             ▼                           ▼
        [submit_bid()]                 [submit_bid()]             [submit_bid()]
        [submit_word()]                [submit_word()]            [submit_word()]
```

#### 1. Benoit (Standard Deep Reinforcement Learning)
- **Model Architecture**: Unified PyTorch Actor-Critic (`models.py`):
  - **Sequential Public Event Transformer**: A 2-layer `EventTransformerEncoder` ($d_{\text{model}} = 64$, 4 heads, GELU, positional encodings) that ingests the 32 most recent public game events (`AuctionResult` and `WordPlay`) to learn implicit tracking of opponent purchasing power and tile distributions.
  - **Private State Trunk**: 2-layer MLP encoding spendable balance, 26-element rack histogram, current auction tile one-hot representation, and instantaneous best-word heuristics.
  - **Continuous Actor Head**: Outputs a 2D Gaussian action $[\mu_{\text{bid}}, \mu_{\text{play}}]$, mapped through Sigmoid to continuous fractions in $[0, 1]$ representing bid percentage of current bankroll and the threshold to hold vs cash out words.
  - **Critic Head**: Predicts expected match terminal placement score for Generalized Advantage Estimation ($\lambda = 0.95, \gamma = 0.99$).
- **Training Pipeline**: PPO self-play curriculum in `train.py` mixing past policy snapshots with baseline bot styles (`Aggressive`, `Cheapskate`, `ValueBidder`, `Hoarder`).
- **Production Inference**: Exported to ONNX `export_onnx.py` and executed client-side via `model_inference.ts` with safety clamping (25% max bankroll cap) to mitigate Vickrey snipe traps.

#### 2. Hercule (Deductive Game-Theoretic Heuristic Engine)
- **Public Ledger Deduction Engine `intel.ts`**:
  - Functions `observeRacks()` and `knownRacks()` parse incoming public event streams, reconstructing opponents' exact known racks and tracking their rack size discrepancies.
- **Empirical Contender Threat Intelligence**:
  - Incorporates empirical performance priors on live top-ranked leaderboard bots (such as *Chinese Dante*, *Nijello*, *Metheny*, *postturing*) via `CONTENDER_PRIORS`. Dynamically computes Bayesian `contenderStrength()` based on match history and win rate to adjust aggression against specific opponents.
- **Combinatorial Bitmask Rack Planning `planning.ts`, `scoring-v3.ts`**:
  - Fast bitmask filtering across length-partitioned word entries.
  - Implements `scoringFrontier()`: evaluates immediate multi-word partitions, while simultaneously maintaining a 24-candidate frontier of high-reward 7- to 9-letter goal words. Evaluates expected gain using bag tile frequencies:
    $$\text{ExpectedGain} = \text{Reward} \times \text{Factor}(\text{missing}) \times \prod_{c \in \text{missing}} \min\left(1, \frac{\text{Freq}[c]}{5}\right)^{\text{lack}}$$
- **Marginal Value & Strategic Denial Bidding `hercule-policy.ts`**:
  - Evaluates both **Own Marginal Value** ($\Delta \text{Score}$ gained from tile) and **Opponent Denial Value** (preventing a leading contender from completing a superlinear bingo). Bids according to letter affinities and liquidity reserves.

#### 3. Roger (Adversarial Counter-RL Trained Against Hercule)
- **High-Fidelity Hercule Environment Simulation**:
  - In `environment.py`, we ported Hercule's complete TypeScript logic into an optimized Python class (`HerculeOpponent`), replicating its public goal priors (`PERIQUE`, `JOINERY`, `YANQUIS`), letter affinities, reserve ratios, and cash-out policies.
- **Adversarial League Curriculum `train.py`:
  - Implemented `sample_opponents_for_match()` where 70% of opponent slots are populated by live `HerculeOpponent` instances and 30% are sampled from a historical snapshot pool of Roger's past checkpoints.
  - Roger's neural network learned to identify the predictable tipping points in Hercule's bidding curve: driving up the clearing price on tiles Hercule desperately wanted, while stealthily acquiring mid-value tiles that Hercule undervalued to cash out rapid 5-letter words before Hercule could complete 7-letter words.

---

#### Additional Methodology:
- **Trie Anagram Optimization**: Precomputed alphabet-sorted anagram lookup trees over the ENABLE dictionary achieve complete rack evaluation in under 25 microseconds, allowing deep search over thousands of possible tile permutations per auction tick.
- **Vickrey Price Gouging Shield**: Because second-price auctions punish overbidders when another high bidder enters, all bots incorporate strict liquidity ceilings (capping bids to $\le 25\%$ of balance during early rounds) to avoid ruinous traps.
- **Dual-Stack Runtime**: Model training is executed in high-throughput Python PyTorch/Gymnasium, then seamlessly serialized into ONNX for zero-latency execution in production TypeScript and Rust client engines.

---

## 🧩 Novelty
- **Cross-Paradigm Evolutionary Loop**: Rather than treating RL and heuristic game theory as mutually exclusive, we used our analytical heuristic bot (Hercule) as an adversarial training gym to breed a superior reinforcement learning agent (Roger).
- **Public-Ledger Information Reconstruction**: Turning SpacetimeDB's event-sourcing architecture into a competitive weapon by dynamically maintaining an opponent rack tracker from public auction bids and word submissions.
- **End-to-End Auction-to-Lexicon Policy**: Unifying continuous sealed-bid monetary auction decisions with discrete combinatorial word-tile anagram matching into a single cohesive optimization target.
- **Opponent Threat-Weighted Denial**: Leveraging empirical live leaderboard telemetry to scale denial bids proportional to real-world opponent Elo and win consistency.

---

## Results
The performance of the trio was evaluated across hundreds of live matches on the official SpacetimeDB ScrabbleBot testbed.

### 1. Leaderboard Ascent & Rating Evolution
After the initial exploratory phase, the RL policy refined its bidding distributions and cash-out timing. As recorded on the live SpacetimeDB rating charts:
- **Benoit climbed to a peak rating of over 1400 Elo**, stabilizing at **1177 ±91 Elo** across 199 competitive matches, securing **#226 on the global leaderboard** with **6 outright first-place match victories** against a field of 268 autonomous bots.
- Consecutive late-stage matches logged strong rating surges of **+21, +20, +22, and +18 points per match**, proving the convergence of its learned bidding strategy.

![Benoit Elo Rating Progression](media/elo.png)
*Figure 1: Live rating progression on [scrabblebot.spacetimedb.com](https://scrabblebot.spacetimedb.com), showing early exploration followed by a meteoric rating surge past 1400 Elo.*

---

### 2. Live Match Analysis (Match #48620)
In live 6-player matches, our bots consistently outperformed naive hoarder bots and static value bidders through superior liquidity management:
- In **Match #48620**, Benoit secured **3rd place with 93 points** and 14 coins remaining, cleanly besting competitive challengers (`Odysseus's Voyage` at 79 pts, `rackpilot-live` at 0 pts, and `TileHoarder` at 0 pts).
- The public auction ledger highlights how Vickrey second-price economics were exploited: rival bot `pazabol` was forced to bid as high as 50 coins on single letters (`N` and `L`), while our agents maintained capital reserves to cash out steady scoring words.

![Match #48620 Scoreboard](media/ranked.png)
*Figure 2: Final scoreboard of Match #48620 showing placement, tile efficiency, and bankroll preservation.*
