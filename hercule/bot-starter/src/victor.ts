import { competitiveBid, chooseCompetitiveWord, type CompetitiveDecision } from "./competition.js";
import { chooseWord, wordReward, type WordContext } from "./strategy.js";
import { planRack, rackSize } from "./planning.js";
import { contenderStrength, type Opponent } from "./intel.js";

export const VICTOR_POLICY_VERSION="victor-hybrid-v1";
export type VictorMode="build"|"defend"|"rebuild";
export interface VictorContext extends WordContext {
  myBalance:number;
  myScore:number;
  opponents:Opponent[];
}
export interface VictorPosition {
  ownProjected:number;
  rivalProjected:number;
  rivalCash:number;
}
export interface VictorDecision extends CompetitiveDecision {
  mode:VictorMode;
  modeReason:string;
  position:VictorPosition;
}
export const VICTOR_SETTINGS={
  earnedPointsToDefend:30,
  entryCash:110,
  relativeEntryCash:80,
  cashAdvantage:1.35,
  exitCash:55,
  entryScoreGap:12,
  exitScoreGap:25,
  defenseReserve:30,
  defensiveExploration:3,
  attackWeight:0.8,
  defenseWeight:1.6,
  pollMs:100,
} as const;

// Forecasts are risk heuristics over public information, not recovered private
// racks or guaranteed final scores. Include broke rivals: a held word can refill
// their cash at any moment.
function positionFor(ctx:VictorContext): VictorPosition {
  const real=ctx.opponents.filter(opponent=>!opponent.isSimulated);
  const forecasts=real.map(opponent=>{
    const word=chooseWord({myRack:opponent.knownRack,dictionary:ctx.dictionary});
    const pending=Math.max(word?wordReward(word):0,
      (opponent.rackSize??rackSize(opponent.knownRack))*2.5*contenderStrength(opponent));
    return opponent.score+pending;
  });
  return {ownProjected:ctx.myScore+planRack(ctx).score,
    rivalProjected:Math.max(0,...forecasts),rivalCash:Math.max(0,...real.map(opponent=>opponent.balance))};
}

export class VictorMateoPolicy {
  private currentMode:VictorMode="build";
  private reason="Opening acquisition: earn a scoring and cash position first.";
  private switches=0;
  get snapshot() {return {mode:this.currentMode,reason:this.reason,transitions:this.switches};}
  reset() {
    this.currentMode="build";this.switches=0;
    this.reason="Opening acquisition: earn a scoring and cash position first.";
  }

  private update(ctx:VictorContext): VictorPosition {
    const position=positionFor(ctx), settings=VICTOR_SETTINGS;
    let next=this.currentMode, reason=this.reason;
    const real=ctx.opponents.some(opponent=>!opponent.isSimulated);
    if(this.currentMode==="defend") {
      if(ctx.myBalance<settings.exitCash) {
        next="rebuild";reason="Cash is below the defense floor; rebuild scoring income.";
      } else if(!real || position.ownProjected+settings.exitScoreGap<position.rivalProjected) {
        next="rebuild";reason="The projected position needs more points before protection.";
      }
    } else {
      const funded=ctx.myBalance>=settings.entryCash ||
        ctx.myBalance>=settings.relativeEntryCash&&ctx.myBalance>=settings.cashAdvantage*position.rivalCash;
      const competitive=position.ownProjected+settings.entryScoreGap>=position.rivalProjected;
      if(real&&ctx.myScore>=settings.earnedPointsToDefend&&funded&&competitive) {
        next="defend";reason="Earned points, a funded budget, and a competitive projected position justify defense.";
      }
    }
    if(next!==this.currentMode) {this.currentMode=next;this.reason=reason;this.switches++;}
    return position;
  }

  bid(ctx:VictorContext&{letter:string}): VictorDecision {
    if(!Number.isFinite(ctx.myBalance)||ctx.myBalance<0||!Number.isFinite(ctx.myScore)) {
      return {amount:0,baseline:0,threat:0,weight:0,mode:this.currentMode,modeReason:this.reason,
        position:{ownProjected:0,rivalProjected:0,rivalCash:0}};
    }
    const position=this.update(ctx), defending=this.currentMode==="defend";
    const decision=competitiveBid(ctx,ctx.opponents,defending?"hercule":"wags",
      defending?VICTOR_SETTINGS.defenseWeight:VICTOR_SETTINGS.attackWeight);
    if(defending) {
      const afterRack=new Map(ctx.myRack);afterRack.set(ctx.letter,(afterRack.get(ctx.letter)??0)+1);
      const before=planRack(ctx), after=planRack({...ctx,myRack:afterRack});
      const usableFuture=Math.max(0,(ctx.remainingTiles??Infinity)-1);
      const value=(plan:ReturnType<typeof planRack>)=>plan.score+
        ((plan.goal?.missing??Infinity)<=usableFuture?(plan.goal?.expectedGain??0):0);
      const ending=(ctx.remainingTiles??Infinity)<=8||(ctx.progress??0)>=0.88;
      const productive=(decision.ownGain??0)>=6 || value(after)-value(before)>=8 ||
        ending&&(decision.ownGain??0)>0;
      const baseline=productive?decision.baseline:Math.min(decision.baseline,VICTOR_SETTINGS.defensiveExploration);
      let pressure=0;
      if(decision.amount>decision.baseline) {
        pressure=decision.confidence==="uncertain"
          ?baseline+Math.min(3,decision.amount-decision.baseline):decision.amount;
      }
      const balance=Math.floor(ctx.myBalance);
      const reserve=ending?0:Math.min(VICTOR_SETTINGS.defenseReserve,Math.floor(balance*0.35));
      decision.baseline=Math.min(baseline,balance-reserve);
      decision.amount=Math.min(balance-reserve,Math.max(decision.baseline,pressure));
    }
    return {...decision,mode:this.currentMode,modeReason:this.reason,position};
  }

  word(ctx:VictorContext): string|null {
    if(!Number.isFinite(ctx.myBalance)||ctx.myBalance<0||!Number.isFinite(ctx.myScore)) return null;
    this.update(ctx);
    // Defense still earns income. The existing planner protects promising
    // longer words, cashes valuable payouts, and flushes before match end.
    return chooseCompetitiveWord(ctx,this.currentMode==="defend"?"hercule":"wags");
  }
}

export function assertVictorIdentity(registeredName:string|undefined): void {
  if(registeredName!=="VictorMateo") throw new Error(`This token belongs to ${registeredName??"an unknown bot"}. VictorMateo needs a token for a separately registered bot named VictorMateo.`);
}
