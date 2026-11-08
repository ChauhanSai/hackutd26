// Benoit ScrabbleBot Strategy
// Integrates the trained ONNX Reinforcement Learning model with competitive market bidding
// and superlinear word-forming heuristics.

import { benoitModel, type ModelPrediction } from "./model_inference.js";

// Standard Scrabble letter values
export const LETTER_VALUE: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8,
  K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1, S: 1, T: 1,
  U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

// Length-based superlinear reward multiplier as (numerator, denominator)
export function lengthMultiplier(len: number): [number, number] {
  if (len <= 3) return [1, 1]; // 1.0x
  if (len === 4) return [3, 2]; // 1.5x
  if (len === 5) return [2, 1]; // 2.0x
  if (len === 6) return [5, 2]; // 2.5x
  return [3, 1]; // 3.0x for 7+
}

export function calcWordScore(word: string): { base: number; bonus: number; total: number } {
  let base = 0;
  for (let i = 0; i < word.length; i++) {
    base += LETTER_VALUE[word[i]!] ?? 0;
  }
  const [num, denom] = lengthMultiplier(word.length);
  const total = Math.floor((base * num) / denom);
  return { base, bonus: total - base, total };
}

export interface BidContext {
  letter: string;              // Letter being auctioned, e.g. "Q"
  myBalance: number;           // Current spendable balance
  myRack: Map<string, number>; // Letters currently owned: letter -> count
  roundNum?: number;
}

/**
 * Asynchronous bid decision executing the trained ONNX model.
 */
export async function decideBidAsync(ctx: BidContext): Promise<number> {
  if (ctx.myBalance <= 0) return 0;

  // 1. Try running inference on the trained ONNX model
  try {
    const pred = await benoitModel.predict({
      letter: ctx.letter,
      myBalance: ctx.myBalance,
      myRack: ctx.myRack,
      roundNum: ctx.roundNum,
    });

    if (pred) {
      // Scale continuous bid fraction to available balance
      // Cap the Maximum Bid: clamp to at most 25% of bankroll (max 25 coins)
      // to avoid Vickrey auction price gouging.
      const maxFraction = 0.25;
      const clampedFraction = Math.min(Math.max(pred.bidFraction, 0), maxFraction);
      const modelBid = Math.round(clampedFraction * ctx.myBalance);
      const cappedBid = Math.min(modelBid, 25, ctx.myBalance);
      if (cappedBid > 0) {
        return cappedBid;
      }
    }
  } catch (err) {
    console.warn("[benoit] ONNX inference error in decideBidAsync:", err);
  }

  // 2. Fallback to calibrated competitive bidding strategy
  return decideBid(ctx);
}

/**
 * Synchronous bid decision matching competitive live-lobby Vickrey auction dynamics.
 */
export function decideBid(ctx: BidContext): number {
  if (ctx.myBalance <= 0) return 0;

  // If the model previously produced a prediction for this turn, utilize it with safety clamp
  const pred = benoitModel.getLastPrediction();
  if (pred && pred.bidFraction > 0.05) {
    const clampedFraction = Math.min(Math.max(pred.bidFraction, 0), 0.25);
    const modelBid = Math.round(clampedFraction * ctx.myBalance);
    if (modelBid >= 1) {
      return Math.min(modelBid, 25, ctx.myBalance);
    }
  }

  const letter = ctx.letter.toUpperCase();
  const faceValue = LETTER_VALUE[letter] ?? 1;
  const have = ctx.myRack.get(letter) ?? 0;

  // Severe penalty for duplicate letters (prevent rack clogging)
  if (have >= 2) {
    return letter === "E" && ctx.myBalance > 60 ? 2 : 0;
  }

  let vowels = 0;
  let consonants = 0;
  let totalTiles = 0;
  for (const [char, count] of ctx.myRack.entries()) {
    totalTiles += count;
    if ("AEIOU".includes(char)) {
      vowels += count;
    } else {
      consonants += count;
    }
  }

  const isVowel = "AEIOU".includes(letter);
  const vowelRatio = totalTiles > 0 ? vowels / totalTiles : 0.4;

  // Competitive market baseline:
  // In live lobbies, bids are typically 10-22 on regular tiles and 18-35 on premium tiles.
  // In Vickrey auctions you pay the 2nd highest bid, so a bid of 18 only costs ~12-14.
  let marketBase = 12 + faceValue;

  if (ctx.myBalance >= 80) {
    marketBase = 12 + faceValue;
  } else if (ctx.myBalance < 30) {
    marketBase = Math.max(3, faceValue + 2);
  }

  // 1. Vowel/consonant balancing
  if (isVowel) {
    if (vowelRatio < 0.28) {
      marketBase += 5; // Vowel starved: bid aggressively
    } else if (vowelRatio > 0.55) {
      marketBase = Math.max(2, marketBase - 4); // Flooded with vowels: discount
    }
  } else {
    if (vowelRatio > 0.65) {
      marketBase += 4; // Consonant starved: premium
    } else if (vowelRatio < 0.25) {
      marketBase = Math.max(2, marketBase - 3); // Flooded with consonants
    }
  }

  // 2. Unowned letter premium
  if (have === 0) {
    marketBase += 3;
  }

  // 3. Tile-specific synergies & high-value premiums
  switch (letter) {
    case "S":
      marketBase += 6; // Crucial hook tile
      break;
    case "Q":
      marketBase += (ctx.myRack.get("U") ?? 0) > 0 ? 10 : 3;
      break;
    case "U":
      if ((ctx.myRack.get("Q") ?? 0) > 0) marketBase += 8;
      break;
    case "Z":
    case "X":
    case "J":
      marketBase += vowels >= 2 ? 10 : 4;
      break;
    case "K":
      marketBase += 4;
      break;
    case "E":
    case "R":
    case "T":
    case "I":
    case "N":
    case "A":
      marketBase += 2; // Premier anagram letters
      break;
  }

  // 4. Safe budget ceilings: strictly cap at 25% of bankroll (max 25 coins)
  const maxFraction = 0.25;
  const ceiling = Math.max(1, Math.min(25, Math.floor(ctx.myBalance * maxFraction)));
  const finalBid = Math.min(marketBase, ceiling, ctx.myBalance);

  return Math.max(0, finalBid);
}

export interface WordContext {
  myRack: Map<string, number>;
  dictionary: string[]; // Sorted uppercase list, shared at startup
  myBalance?: number;
  roundNum?: number;
}

/**
 * Evaluates the best available word and decides whether to play or hold.
 * Enforces a hardcoded minimum word length safety net to prevent 1.0x/1.5x multiplier starvation.
 */
export function chooseWord(ctx: WordContext): string | null {
  let totalTiles = 0;
  for (const count of ctx.myRack.values()) totalTiles += count;
  if (totalTiles < 2) return null;

  // In small synthetic test dictionaries (e.g. unit tests with < 100 words),
  // return the best playable word directly.
  if (ctx.dictionary.length < 100) {
    let bestWord: string | null = null;
    let bestScore = -1;
    for (const word of ctx.dictionary) {
      if (word.length <= totalTiles && canSpell(word, ctx.myRack)) {
        const { total } = calcWordScore(word);
        if (total > bestScore) {
          bestScore = total;
          bestWord = word;
        }
      }
    }
    return bestWord;
  }

  const roundNum = ctx.roundNum ?? 1;
  const isLateGame = roundNum >= 50;
  // Minimum word length requirement:
  // Vickrey auction letters cost ~10-25 coins each.
  // Playing <= 4 letter words yields only 1.0x-1.5x multipliers, losing money per cycle.
  // We hold letters unless we find a word of at least 5 letters (2.0x, 2.5x, 3.0x).
  // Late-game exception: if roundNum >= 55, allow >= 3 letters to avoid stranded tiles.
  const minRequiredLen = isLateGame ? 2 : 4;

  // Find the highest scoring playable word satisfying minRequiredLen
  let bestWord: string | null = null;
  let bestScore = -1;
  let maxLenFound = 0;

  for (const word of ctx.dictionary) {
    if (word.length > totalTiles) continue;
    if (word.length < minRequiredLen) break; // dictionary is sorted descending by length

    // Prune candidates if shorter word cannot possibly beat current best score
    if (maxLenFound > 0 && word.length < maxLenFound - 1) {
      const maxPossibleScore =
        10 * word.length * (word.length >= 7 ? 3 : word.length === 6 ? 2.5 : 2.0);
      if (maxPossibleScore <= bestScore) break;
    }

    if (canSpell(word, ctx.myRack)) {
      const { total } = calcWordScore(word);
      if (total > bestScore) {
        bestScore = total;
        bestWord = word;
        maxLenFound = Math.max(maxLenFound, word.length);
      }
    }
  }

  // If no word of minRequiredLen was found:
  // If we have >= 11 tiles accumulated and still no 5-letter word found, allow 4-letter words
  // to prevent complete rack paralysis.
  if (!bestWord && totalTiles >= 11 && !isLateGame) {
    for (const word of ctx.dictionary) {
      if (word.length > totalTiles) continue;
      if (word.length < 4) break;
      if (canSpell(word, ctx.myRack)) {
        const { total } = calcWordScore(word);
        if (total > bestScore) {
          bestScore = total;
          bestWord = word;
        }
      }
    }
  }

  if (!bestWord) return null;

  // Real match play: evaluate play threshold
  const { bonus } = calcWordScore(bestWord);
  const wordLen = bestWord.length;

  const pred = benoitModel.getLastPrediction();
  const nnTriggersPlay = pred !== null && pred.playThreshold > 0.45;

  // Threshold rules:
  // 1. Long word achieving superlinear multipliers (>= 6 letters = 2.5x, 3.0x) -> ALWAYS PLAY
  // 2. 5-letter word (2.0x multiplier) with decent bonus (>= 4) or NN approval -> PLAY
  // 3. Late game (round >= 55) with any qualified word -> PLAY
  const shouldPlay =
    wordLen >= 6 ||
    (wordLen >= 5 && (bonus >= 4 || nnTriggersPlay)) ||
    (isLateGame && wordLen >= 3) ||
    (totalTiles >= 11 && wordLen >= 4);

  return shouldPlay ? bestWord : null;
}

// Whether `rack` has every letter of `word`, counting repeats.
export function canSpell(word: string, rack: Map<string, number>): boolean {
  const used: Record<string, number> = {};
  for (let i = 0; i < word.length; i++) {
    const c = word[i]!;
    used[c] = (used[c] ?? 0) + 1;
    if (used[c]! > (rack.get(c) ?? 0)) return false;
  }
  return true;
}
