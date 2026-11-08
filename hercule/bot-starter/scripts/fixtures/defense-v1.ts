import { completionValue, decideBid, type BidContext } from "../../src/strategy.js";

export interface PublicRackEvent {
  time: bigint;
  kind: "purchase" | "word";
  botId: string;
  text: string;
}

// These are lower bounds, not complete racks: initial dealt tiles stay private.
export function knownRacks(events: PublicRackEvent[]): Map<string, Map<string, number>> {
  const racks = new Map<string, Map<string, number>>();
  const ordered = [...events].sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1
    : a.kind === b.kind ? 0 : a.kind === "purchase" ? -1 : 1);
  for (const event of ordered) {
    let rack = racks.get(event.botId);
    if (!rack) racks.set(event.botId, rack = new Map());
    for (const letter of event.text) {
      rack.set(letter, event.kind === "purchase" ? (rack.get(letter) ?? 0) + 1
        : Math.max(0, (rack.get(letter) ?? 0) - 1));
    }
  }
  return racks;
}

export interface Opponent {
  id: string;
  name: string;
  score: number;
  balance: number;
  isSimulated: boolean;
  knownRack: Map<string, number>;
}

export interface DefensiveDecision {
  amount: number;
  baseline: number;
  threat: number;
  target?: string;
  targetName?: string;
  weight: number;
}

export function defensiveBid(ctx: BidContext, opponents: Opponent[], weight: number): DefensiveDecision {
  const baseline = decideBid(ctx);
  const result: DefensiveDecision = { amount: baseline, baseline, threat: 0, weight };
  if (weight <= 0 || ctx.myBalance <= 20) return result;
  const real = opponents.filter((opponent) => !opponent.isSimulated && opponent.balance >= 1);
  const leaderScore = Math.max(1, ...real.map((opponent) => opponent.score));
  let bestValue = 0;
  for (const opponent of real) {
    const threat = completionValue({ myRack: opponent.knownRack, dictionary: ctx.dictionary }, ctx.letter);
    // Only pay extra to block a material, observable word completion.
    if (threat < 10) continue;
    const priority = 0.5 + 0.5 * Math.max(0, opponent.score) / leaderScore;
    const value = threat * priority;
    if (value > bestValue) {
      bestValue = value;
      result.threat = threat;
      result.target = opponent.id;
      result.targetName = opponent.name;
    }
  }
  const defensiveBudget = Math.max(0, Math.min(Math.floor(ctx.myBalance * 0.35), Math.floor(ctx.myBalance - 20)));
  result.amount = Math.max(baseline, Math.min(defensiveBudget, Math.floor(bestValue * weight)));
  return result;
}

