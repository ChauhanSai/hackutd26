import { chooseWord, chooseWordAtLeast } from "./strategy.js";

export const BID_FEATURES = ["intercept", ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map((letter) => `letter_${letter}_inverse_balance`),
  "log_balance", "rack_size", "known_copies", "match_progress", "own_score", "leader_gap", "first_price"];

export interface PublicBidState {
  letter: string;
  balance: number;
  score: number;
  rackSize: number;
  knownCopies: number;
  progress: number;
  leaderScore: number;
  firstPrice?: boolean;
}
export interface BidSample extends PublicBidState {
  matchId: number;
  kind: "exact" | "upper" | "lower";
  amount: number;
}
export interface BidFit {
  family: "censored-linear" | "letter-median";
  features: string[];
  coefficients: number[];
  letterFractions: Record<string, number>;
  fallbackFraction: number;
}
export interface PublicOpponentModel {
  schemaVersion: 1;
  name: string;
  botId: number;
  version: number;
  collectedAt: string;
  tier: string;
  leaderboard: { matches: number; wins: number; winRate: number; interval95: [number, number] };
  bid: BidFit;
  word: { minimumLength: number; trainingLengthHistogram: Record<string, number>;
    cashOutBalance: number; flushRemaining: number; rackLimit: number; fallbackIsHeuristic: true };
  evidence: Record<string, unknown>;
}

const clamp = (value: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, value));
export function bidFeatures(state: PublicBidState): number[] {
  const features = [1, ...Array<number>(26).fill(0)];
  const letter = state.letter.charCodeAt(0) - 65;
  if (letter >= 0 && letter < 26) features[letter + 1] = Math.min(5, 100 / Math.max(1, state.balance));
  features.push(Math.log1p(Math.max(0, state.balance)) / Math.log(501), clamp(state.rackSize / 20, 0, 2),
    clamp(state.knownCopies / 4, 0, 1), clamp(state.progress, 0, 1), clamp(state.score / 300, 0, 2),
    clamp((state.leaderScore - state.score) / 200, -2, 2), state.firstPrice ? 1 : 0);
  return features;
}

export function quantile(values: number[], fraction: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(fraction * (sorted.length - 1))))];
}

export function wilsonInterval(wins: number, matches: number): [number, number] {
  if (matches <= 0) return [0, 1];
  const p = wins / matches, z = 1.96, denominator = 1 + z * z / matches;
  const center = (p + z * z / (2 * matches)) / denominator;
  const half = z * Math.sqrt(p * (1 - p) / matches + z * z / (4 * matches * matches)) / denominator;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

// Match IDs, never individual auctions, determine train/test membership.
export function chronologicalSplit(matchIds: number[], fraction = 0.2): { train: number[]; test: number[] } {
  const ids = [...new Set(matchIds)].sort((a, b) => a - b);
  const testCount = Math.min(ids.length - 1, Math.max(1, Math.ceil(ids.length * fraction)));
  return { train: ids.slice(0, ids.length - testCount), test: ids.slice(ids.length - testCount) };
}

export function fitBid(samples: BidSample[], family: BidFit["family"] = "censored-linear"): BidFit {
  const exact = samples.filter((sample) => sample.kind === "exact" && sample.balance > 0);
  const fractions = exact.map((sample) => sample.amount / sample.balance);
  const letterFractions: Record<string, number> = {};
  for (const letter of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
    const letterSamples = exact.filter((sample) => sample.letter === letter).map((sample) => sample.amount / sample.balance);
    if (letterSamples.length >= 3) letterFractions[letter] = quantile(letterSamples, 0.5);
  }
  const fallbackFraction = quantile(fractions, 0.5);
  const coefficients = Array<number>(BID_FEATURES.length).fill(0);
  coefficients[0] = fallbackFraction;
  const model: BidFit = { family, features: BID_FEATURES, coefficients, letterFractions, fallbackFraction };
  if (family === "letter-median" || !samples.length) return model;
  const rows = samples.filter((sample) => sample.balance > 0).map((sample) => ({
    x: bidFeatures(sample), y: clamp(sample.amount / sample.balance, 0, 1), kind: sample.kind,
  }));
  const m = coefficients.map(() => 0), v = coefficients.map(() => 0);
  let step = 0, random = 8297;
  for (let epoch = 0; epoch < 40; epoch++) {
    for (let offset = 0; offset < rows.length; offset += 128) {
      const gradient = coefficients.map(() => 0);
      const size = Math.min(128, rows.length - offset);
      for (let j = 0; j < size; j++) {
        random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
        const row = rows[random % rows.length];
        let prediction = 0;
        for (let k = 0; k < coefficients.length; k++) prediction += coefficients[k] * row.x[k];
        let error = prediction - row.y;
        if (row.kind === "upper") error = Math.max(0, error);
        if (row.kind === "lower") error = Math.min(0, error);
        const weight = row.kind === "exact" ? 3 : 1;
        for (let k = 0; k < coefficients.length; k++) gradient[k] += 2 * weight * error * row.x[k] / size;
      }
      step++;
      for (let k = 0; k < coefficients.length; k++) {
        const g = gradient[k] + 0.0002 * coefficients[k];
        m[k] = 0.9 * m[k] + 0.1 * g;
        v[k] = 0.999 * v[k] + 0.001 * g * g;
        coefficients[k] -= 0.008 * (m[k] / (1 - 0.9 ** step)) / (Math.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8);
      }
    }
  }
  return model;
}

export function predictBid(model: BidFit, state: PublicBidState): number {
  if (state.balance < 1 || !/^[A-Z]$/.test(state.letter)) return 0;
  const features = bidFeatures(state);
  const fraction = model.family === "letter-median" ? model.letterFractions[state.letter] ?? model.fallbackFraction
    : features.reduce((sum, value, i) => sum + value * model.coefficients[i], 0);
  return Math.min(Math.floor(state.balance), Math.max(0, Math.round(clamp(fraction, 0, 1) * state.balance)));
}

export function evaluateBid(model: BidFit, samples: BidSample[]) {
  const exact = samples.filter((sample) => sample.kind === "exact");
  const upper = samples.filter((sample) => sample.kind === "upper");
  let absoluteError = 0, fractionError = 0, upperExcess = 0, violations = 0;
  for (const sample of exact) {
    const error = Math.abs(predictBid(model, sample) - sample.amount);
    absoluteError += error;
    fractionError += error / Math.max(1, sample.balance);
  }
  for (const sample of upper) {
    const excess = Math.max(0, predictBid(model, sample) - sample.amount);
    upperExcess += excess / Math.max(1, sample.balance);
    if (excess > 0) violations++;
  }
  return { exactWinningBids: exact.length, losingBounds: upper.length,
    winningBidMAE: exact.length ? absoluteError / exact.length : null,
    upperViolationRate: upper.length ? violations / upper.length : null,
    selectionLoss: fractionError / Math.max(1, exact.length) + 0.25 * upperExcess / Math.max(1, upper.length) };
}

export interface SampleBotContext {
  myRack: Map<string, number>;
  myBalance: number;
  myScore: number;
  dictionary: readonly string[];
  // Retained public purchases, excluding the hidden initial deal.
  publicKnownRack?: Map<string, number>;
  opponents: { score: number }[];
  progress: number;
  remainingTiles: number;
  auctionType?: "Vickrey" | "FirstPrice";
}

export function createSampleBot(model: PublicOpponentModel) {
  return {
    name: `${model.name} (public-data model)`,
    decideBid(ctx: SampleBotContext & { letter: string }): number {
      return predictBid(model.bid, { letter: ctx.letter, balance: ctx.myBalance, score: ctx.myScore,
        rackSize: [...ctx.myRack.values()].reduce((a,b) => a+b, 0),
        knownCopies: ctx.publicKnownRack?.get(ctx.letter) ?? 0, progress: ctx.progress,
        leaderScore: Math.max(ctx.myScore, ...ctx.opponents.map((opponent) => opponent.score)), firstPrice: ctx.auctionType === "FirstPrice" });
    },
    chooseWord(ctx: SampleBotContext): string | null {
      const rackSize = [...ctx.myRack.values()].reduce((a,b) => a+b, 0);
      const flush = ctx.myBalance <= model.word.cashOutBalance || ctx.remainingTiles <= model.word.flushRemaining || rackSize >= model.word.rackLimit;
      return flush ? chooseWord(ctx) : chooseWordAtLeast(ctx, model.word.minimumLength);
    },
  };
}
