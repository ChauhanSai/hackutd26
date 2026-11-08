import { prepareDictionary, type WordContext } from "./strategy.js";
import { type CompetitiveBidContext } from "./competition.js";
import {scoringBid,chooseScoringWord,SCORING_SETTINGS} from "./scoring-v3.js";

export const WAGS_SETTINGS = {
  commonLetterBid: SCORING_SETTINGS.wags.common, otherLetterBid: SCORING_SETTINGS.wags.other,
  rareLetterBid: SCORING_SETTINGS.wags.rare, maximumBalanceFraction: SCORING_SETTINGS.wags.fraction,
  reserveWhenFunded: SCORING_SETTINGS.wags.reserve, pollMs:100,
} as const;

export function prepareWagsDictionary(dictionary: readonly string[]): void { prepareDictionary(dictionary); }
export function decideWagsDecision(ctx: CompetitiveBidContext) {
  return scoringBid(ctx,ctx.opponents??[],"wags",1);
}
export function decideWagsBid(ctx: CompetitiveBidContext): number { return decideWagsDecision(ctx).amount; }
export function chooseWagsWord(ctx: WordContext): string|null { return chooseScoringWord(ctx,"wags"); }
export function assertWagsIdentity(registeredName: string|undefined): void {
  if (registeredName!=="Wags") throw new Error(`This token belongs to ${registeredName??"an unknown bot"}. Wags needs a token for a separately registered bot named Wags.`);
}
