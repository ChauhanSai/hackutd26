// Score actual word rewards and value letters that complete words.
export const LETTER_VALUE: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8,
  K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1, S: 1, T: 1,
  U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

export interface WordContext {
  myRack: Map<string, number>;
  dictionary: readonly string[];
  myBalance?: number;
  myScore?: number;
  progress?: number;
  // Tiles still to be auctioned after the current auction. Zero is the last tile.
  remainingTiles?: number;
  auctionType?: "Vickrey" | "FirstPrice";
  auctionLetter?: string;
  committedBid?: number;
}

export interface BidContext extends WordContext {
  letter: string;
  myBalance: number;
}

export interface WordEntry {
  word: string;
  reward: number;
  mask: number;
  // Alternating letter index and required count, for letters present in the word.
  letters: Uint8Array;
}

interface RackAnalysis {
  bestWord: string | null;
  bestScore: number;
  completionRewards: Float64Array;
  bestByLength: Map<number, { word: string; reward: number }>;
}

interface DictionaryIndex {
  byLength: WordEntry[][];
  racks: Map<string, RackAnalysis>;
}

const indexes = new WeakMap<readonly string[], DictionaryIndex>();

// Matches spacetimedb/src/letters.ts, including integer rounding.
export function wordReward(word: string): number {
  let base = 0;
  for (const letter of word) base += LETTER_VALUE[letter] ?? 0;
  const multiplier = word.length <= 3 ? 1 : word.length === 4 ? 1.5
    : word.length === 5 ? 2 : word.length === 6 ? 2.5 : 3;
  return Math.floor(base * multiplier);
}

function dictionaryIndex(dictionary: readonly string[]): DictionaryIndex {
  const existing = indexes.get(dictionary);
  if (existing) return existing;
  const index: DictionaryIndex = { byLength: [], racks: new Map() };
  for (const word of dictionary) {
    if (!/^[A-Z]{2,}$/.test(word)) continue;
    const counts = new Uint8Array(26);
    for (const letter of word) counts[letter.charCodeAt(0) - 65]++;
    const letters: number[] = [];
    let mask = 0;
    for (let i = 0; i < 26; i++) if (counts[i]) { letters.push(i, counts[i]); mask |= 1 << i; }
    (index.byLength[word.length] ??= []).push({
      word, reward: wordReward(word), mask, letters: Uint8Array.from(letters),
    });
  }
  indexes.set(dictionary, index);
  return index;
}

export function prepareDictionary(dictionary: readonly string[]): void {
  dictionaryIndex(dictionary);
}

// Share the prepared index with the planners without duplicating the dictionary.
export function dictionaryWordsByLength(dictionary: readonly string[]): readonly (readonly WordEntry[])[] {
  return dictionaryIndex(dictionary).byLength;
}

function analyzeRack(ctx: WordContext): RackAnalysis {
  const index = dictionaryIndex(ctx.dictionary);
  const counts = new Uint16Array(26);
  let total = 0;
  let mask = 0;
  for (let i = 0; i < 26; i++) {
    counts[i] = ctx.myRack.get(String.fromCharCode(65 + i)) ?? 0;
    total += counts[i];
    if (counts[i]) mask |= 1 << i;
  }
  const key = counts.join(",");
  const cached = index.racks.get(key);
  if (cached) return cached;
  const result: RackAnalysis = {
    bestWord: null, bestScore: 0, completionRewards: new Float64Array(26), bestByLength: new Map(),
  };
  const maxLength = Math.min(total + 1, index.byLength.length - 1);
  for (let length = 2; length <= maxLength; length++) {
    for (const entry of index.byLength[length] ?? []) {
      const missingKinds = entry.mask & ~mask;
      if ((missingKinds & (missingKinds - 1)) !== 0) continue;
      let missing = 0;
      let missingLetter = -1;
      for (let i = 0; i < entry.letters.length; i += 2) {
        const letter = entry.letters[i];
        const deficit = entry.letters[i + 1] - counts[letter];
        if (deficit > 0) {
          missing += deficit;
          missingLetter = letter;
          if (missing > 1) break;
        }
      }
      if (missing === 0) {
        const prior = result.bestByLength.get(length);
        if (!prior || entry.reward > prior.reward) result.bestByLength.set(length, { word: entry.word, reward: entry.reward });
      }
      if (missing === 0 && (
        entry.reward > result.bestScore ||
        (entry.reward === result.bestScore && length > (result.bestWord?.length ?? 0))
      )) {
        result.bestWord = entry.word;
        result.bestScore = entry.reward;
      } else if (missing === 1) {
        result.completionRewards[missingLetter] = Math.max(
          result.completionRewards[missingLetter], entry.reward,
        );
      }
    }
  }
  // Bound the cache when the process runs across many matches.
  if (index.racks.size >= 128) index.racks.delete(index.racks.keys().next().value!);
  index.racks.set(key, result);
  return result;
}

export function decideBid(ctx: BidContext): number {
  if (!Number.isFinite(ctx.myBalance) || ctx.myBalance < 1 || !/^[A-Z]$/.test(ctx.letter)) {
    return 0;
  }
  const analysis = analyzeRack(ctx);
  const completion = analysis.completionRewards[ctx.letter.charCodeAt(0) - 65];
  const improvement = Math.max(0, completion - analysis.bestScore);
  const have = ctx.myRack.get(ctx.letter) ?? 0;
  // Cheap exploration favors common letters and avoids stockpiling duplicates.
  const exploration = have >= 2 ? 0 : have === 1 ? 1
    : "AEINORST".includes(ctx.letter) ? 3
    : "JQXZK".includes(ctx.letter) ? 1 : 2;
  const want = Math.max(exploration, Math.floor(improvement * 0.65));
  const budget = Math.max(1, Math.floor(ctx.myBalance * 0.35));
  return Math.min(want, budget, Math.floor(ctx.myBalance));
}

export function chooseWord(ctx: WordContext): string | null {
  return analyzeRack(ctx).bestWord;
}

export function chooseWordAtLeast(ctx: WordContext, minimumLength: number): string | null {
  let best: { word: string; reward: number } | undefined;
  for (const [length, entry] of analyzeRack(ctx).bestByLength) {
    if (length >= minimumLength && (!best || entry.reward > best.reward ||
      (entry.reward === best.reward && length > best.word.length))) best = entry;
  }
  return best?.word ?? null;
}

// Additional best-word reward unlocked by buying one particular letter.
export function completionValue(ctx: WordContext, letter: string): number {
  if (!/^[A-Z]$/.test(letter)) return 0;
  const analysis = analyzeRack(ctx);
  return Math.max(0, analysis.completionRewards[letter.charCodeAt(0) - 65] - analysis.bestScore);
}
