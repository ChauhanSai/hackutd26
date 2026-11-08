import { LETTER_VALUE, chooseWord, wordReward, completionValue, type BidContext, type WordContext } from "./strategy.js";
import { planRack, rackSize, choosePlannedWord } from "./planning.js";
import { contenderStrength, type Opponent } from "./intel.js";

export type PlayingStyle="wags"|"hercule";
export interface CompetitiveBidContext extends BidContext { opponents?: Opponent[] }
export interface CompetitiveDecision {
  amount:number;
  baseline:number;
  threat:number;
  target?:string;
  targetName?:string;
  weight:number;
  confidence?:"exact"|"partial"|"uncertain";
  ownGain?:number;
}
export const PLAY_STYLES={
  wags:{common:18,other:12,rare:10,fraction:0.75,reserve:4,gain:1.15,denial:0.7},
  hercule:{common:11,other:8,rare:6,fraction:0.45,reserve:12,gain:0.9,denial:1.0},
} as const;

function withLetter(ctx: WordContext, letter:string): WordContext {
  const myRack=new Map(ctx.myRack);myRack.set(letter,(myRack.get(letter)??0)+1);
  return {...ctx,myRack};
}
export function competitiveBid(ctx: BidContext, opponents: Opponent[], style: PlayingStyle, defenseWeight=1): CompetitiveDecision {
  const result:CompetitiveDecision={amount:0,baseline:0,threat:0,weight:defenseWeight,ownGain:0};
  if (!Number.isFinite(ctx.myBalance)||ctx.myBalance<1||!/^[A-Z]$/.test(ctx.letter)||(ctx.remainingTiles??Infinity)<=0) return result;
  const settings=PLAY_STYLES[style], balance=Math.floor(ctx.myBalance), size=rackSize(ctx.myRack);
  const before=planRack(ctx), after=planRack(withLetter(ctx,ctx.letter));
  const usableFuture=Math.max(0,(ctx.remainingTiles??Infinity)-1);
  const projectedValue=(plan:ReturnType<typeof planRack>)=>plan.score+((plan.goal?.missing??Infinity)<=usableFuture?(plan.goal?.expectedGain??0):0);
  const gain=Math.max(0,after.score-before.score), potential=Math.max(0,projectedValue(after)-projectedValue(before));
  result.ownGain=gain;
  const have=ctx.myRack.get(ctx.letter)??0, common="AEIOUNRSTL".includes(ctx.letter), rare=(LETTER_VALUE[ctx.letter]??0)>=8;
  let explore:number=common?settings.common:rare?settings.rare:settings.other;
  if (have>=2) explore=Math.floor(explore*0.3);
  const vowels=[...ctx.myRack].reduce((s,[letter,n])=>s+("AEIOU".includes(letter)?n:0),0);
  if (size>=4 && (vowels/size<0.25&&"AEIOU".includes(ctx.letter)||vowels/size>0.55&&!"AEIOU".includes(ctx.letter))) explore+=3;
  if (size>=10 && gain===0 && potential<2) explore=Math.min(explore,2);
  const ending=(ctx.remainingTiles??Infinity)<=8 || (ctx.progress??0)>=0.88;
  if (ending && gain===0) explore=0; // A stranded tile cannot turn cash into points.
  const reserve=ending?0:Math.min(settings.reserve,Math.floor(balance*0.15));
  const fraction=ending?(style==="wags"?0.95:0.8):settings.fraction;
  const budget=Math.max(0,Math.min(balance-reserve,Math.max(1,Math.floor(balance*fraction))));
  const desire=Math.max(explore,Math.ceil(settings.gain*Math.max(gain,potential)+2*(gain>0?1:0)));
  result.amount=result.baseline=Math.min(budget,desire);
  if (!Number.isFinite(defenseWeight)||defenseWeight<=0 || !budget) return result;

  const real=opponents.filter(opponent=>!opponent.isSimulated&&opponent.balance>=1);
  // Opponent completion analysis computes all 26 marginal letters in one
  // cached pass. Reserve multi-word packing and future-goal search for us.
  const projected=(opponent:Opponent)=>{
    const word=chooseWord({myRack:opponent.knownRack,dictionary:ctx.dictionary});
    return opponent.score+Math.max(word?wordReward(word):0,(opponent.rackSize??rackSize(opponent.knownRack))*2);
  };
  const highest=Math.max(1,...real.map(projected));
  let bestPressure=0;
  for (const opponent of real) {
    const known=rackSize(opponent.knownRack), total=opponent.rackSize??known;
    const exact=opponent.rackSize!==undefined&&total===known;
    const own={myRack:opponent.knownRack,dictionary:ctx.dictionary};
    const threat=completionValue(own,ctx.letter);
    const strength=contenderStrength(opponent);
    const priority=strength*(0.7+0.3*projected(opponent)/highest);
    // A lower-bound rack is not a complete private rack. Discount its marginal
    // value; small capped pressure acknowledges unknown starting tiles without
    // manufacturing a supposed word completion from them.
    const confidence=exact?1:0.65;
    const credible=threat>=(exact?8:12)?threat*confidence:0;
    const insurance=!exact&&total>=5&&total-known>0&&common&&strength>1.1
      ?Math.min(4,(total-known)*0.6)*Math.min(1,(ctx.remainingTiles??30)/12):0;
    // Own completion and denial values are alternative reasons to buy the
    // same tile. Do not blindly add a full block valuation to the opening bid.
    const scale=priority*Math.min(1.6,defenseWeight);
    const pressure=Math.max(credible*settings.denial*scale,insurance>0?result.baseline+insurance*scale:0);
    if (pressure>bestPressure) {
      bestPressure=pressure;result.threat=threat;result.target=opponent.id;result.targetName=opponent.name;
      result.confidence=credible>0?(exact?"exact":"partial"):"uncertain";
    }
  }
  // A block must remain fundable. Defensive mode leaves more cash for the next
  // completion; offensive mode tolerates a larger commitment to take the tile.
  const extraCap=Math.floor(balance*(style==="wags"?0.35:0.3));
  let extra=Math.min(extraCap,Math.max(0,Math.floor(bestPressure)-result.baseline));
  if (ctx.auctionType==="FirstPrice") extra=Math.floor(extra*0.7);
  result.amount=Math.min(budget,result.baseline+extra);
  return result;
}

export function chooseCompetitiveWord(ctx: WordContext, style:PlayingStyle) {
  return choosePlannedWord(ctx,style);
}
