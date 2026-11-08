import { type BidContext, type WordContext } from "./strategy.js";
import { type CompetitiveDecision } from "./competition.js";
import {scoringBid,chooseScoringWord} from "./scoring-v3.js";
import { type Opponent } from "./intel.js";
export { knownRacks, observeRacks, type Opponent, type PublicRackEvent } from "./intel.js";
export type DefensiveDecision=CompetitiveDecision;

export function defensiveBid(ctx: BidContext, opponents: Opponent[], weight: number): DefensiveDecision {
  return scoringBid(ctx,opponents,"hercule",weight);
}
export function chooseHerculeWord(ctx: WordContext): string|null { return chooseScoringWord(ctx,"hercule"); }
