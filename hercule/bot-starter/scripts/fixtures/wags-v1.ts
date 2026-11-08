import { chooseWord, LETTER_VALUE, wordReward, type BidContext, type WordContext } from "../../src/strategy.js";

interface ShortWord { word: string; reward: number; letters: Uint8Array }
interface ShortIndex { words: ShortWord[]; racks: Map<string, { word: string | null; reward: number }> }
const indexes = new WeakMap<readonly string[], ShortIndex>();

export const WAGS_SETTINGS = {
  commonLetterBid: 12, otherLetterBid: 8, rareLetterBid: 10,
  maximumBalanceFraction: 0.6, reserveWhenFunded: 5,
  minimumWordLength: 2, maximumPreferredWordLength: 4, pollMs: 100,
} as const;

export function prepareWagsDictionary(dictionary: readonly string[]): void {
  if (indexes.has(dictionary)) return;
  const words = dictionary.filter((word) => /^[A-Z]{2,4}$/.test(word)).map((word) => {
    const counts = new Uint8Array(26);
    for (const letter of word) counts[letter.charCodeAt(0) - 65]++;
    const letters: number[] = [];
    for (let i = 0; i < 26; i++) if (counts[i]) letters.push(i, counts[i]);
    return { word, reward: wordReward(word), letters: Uint8Array.from(letters) };
  }).sort((a,b) => b.reward / b.word.length - a.reward / a.word.length || a.word.length-b.word.length || b.reward-a.reward || a.word.localeCompare(b.word));
  indexes.set(dictionary, { words, racks: new Map() });
}

function shortWords(ctx: WordContext) {
  prepareWagsDictionary(ctx.dictionary);
  const index = indexes.get(ctx.dictionary)!;
  const counts = Array.from({ length: 26 }, (_, i) => ctx.myRack.get(String.fromCharCode(65+i)) ?? 0);
  const key = counts.join(",");
  const cached = index.racks.get(key);
  if (cached) return cached;
  let word: string | null = null;
  let reward = 0;
  for (const entry of index.words) {
    let canSpell = true;
    for (let i = 0; i < entry.letters.length; i += 2) {
      if (counts[entry.letters[i]] < entry.letters[i+1]) { canSpell = false; break; }
    }
    if (!canSpell) continue;
    word ??= entry.word; // Sorted by reward per tile, then shortest on ties.
    reward = Math.max(reward,entry.reward);
  }
  const result = { word, reward };
  if (index.racks.size >= 256) index.racks.delete(index.racks.keys().next().value!);
  index.racks.set(key,result);
  return result;
}

export function decideWagsBid(ctx: BidContext): number {
  if (!Number.isFinite(ctx.myBalance) || ctx.myBalance < 1 || !/^[A-Z]$/.test(ctx.letter)) return 0;
  const have = ctx.myRack.get(ctx.letter) ?? 0;
  const base = "AEIOUNRSTL".includes(ctx.letter) ? WAGS_SETTINGS.commonLetterBid
    : (LETTER_VALUE[ctx.letter] ?? 0) >= 8 ? WAGS_SETTINGS.rareLetterBid : WAGS_SETTINGS.otherLetterBid;
  const currentReward = shortWords(ctx).reward;
  const addedRack = new Map(ctx.myRack);
  addedRack.set(ctx.letter,have+1);
  const completionGain = Math.max(0,shortWords({ ...ctx,myRack:addedRack }).reward-currentReward);
  const exploration = have >= 2 ? Math.max(3,Math.ceil(base/2)) : base;
  const want = Math.max(exploration,completionGain > 0 ? Math.ceil(1.25*completionGain+6) : 0);
  const balance = Math.floor(ctx.myBalance);
  const reserve = balance > 10 ? WAGS_SETTINGS.reserveWhenFunded : 0;
  const budget = Math.min(balance-reserve,Math.max(1,Math.floor(balance*WAGS_SETTINGS.maximumBalanceFraction)));
  return Math.max(0,Math.min(want,budget));
}

export function chooseWagsWord(ctx: WordContext): string | null {
  // Cash short words immediately. A long-only rack can still be liquidated.
  return shortWords(ctx).word ?? chooseWord(ctx);
}

export function assertWagsIdentity(registeredName: string | undefined): void {
  if (registeredName !== "Wags") throw new Error(`This token belongs to ${registeredName ?? "an unknown bot"}. Wags needs a token for a separately registered bot named Wags.`);
}

