import {scoringBid,chooseScoringWord,SCORING_SETTINGS,type ScoringDecision} from './scoring-v3.js';
import {type BidContext,type WordContext} from './strategy.js';
import {HERCULE_WORD_PRIORS,HERCULE_LETTER_AFFINITY} from './hercule-priors.js';

export const HERCULE_POLICY_VERSION='hercule-offense-v4';
export interface HerculeConfig {id:string;mode:'none'|'spending'|'public';maxExtra:number}
export const HERCULE_OFFENSE_VARIANTS:readonly HerculeConfig[]=[
  {id:'offense-only',mode:'none',maxExtra:0},
  {id:'offense-spend-six',mode:'spending',maxExtra:6},
  {id:'public-three',mode:'public',maxExtra:3},
  {id:'public-six',mode:'public',maxExtra:6},
];
const FREQUENCY=[9,2,2,4,12,2,3,2,9,1,1,4,2,6,8,2,1,6,4,6,4,2,2,1,2,1];
const indexes=new WeakMap<readonly string[],Set<string>>();
interface PublicGoal {word:string;reward:number;missing:number;missingLetters:string[];expectedGain:number;games:number;bots:number}
// The full v3 planner remains the acquisition floor and word policy. Public
// preferences only authorize a small additional own-value bid, never denial.
export function publicHerculeGoal(ctx:WordContext):PublicGoal|undefined {
  let dictionary=indexes.get(ctx.dictionary);if(!dictionary){dictionary=new Set(ctx.dictionary);indexes.set(ctx.dictionary,dictionary);}
  const remaining=ctx.remainingTiles??Infinity;
  let best:PublicGoal|undefined;
  for(const p of HERCULE_WORD_PRIORS){
    if(!dictionary.has(p.word))continue;
    const need=new Map<string,number>();for(const c of p.word)need.set(c,(need.get(c)??0)+1);
    const missingLetters:string[]=[];let likelihood=1,held=0,possible=true;
    for(const [c,n] of need){const index=c.charCodeAt(0)-65,lack=Math.max(0,n-(ctx.myRack.get(c)??0));if(n>FREQUENCY[index]){possible=false;break;}
      held+=Math.min(n,ctx.myRack.get(c)??0);for(let k=0;k<lack;k++)missingLetters.push(c);likelihood*=Math.min(1,FREQUENCY[index]/5)**lack;}
    const missing=missingLetters.length;if(!possible||held<4||missing>3||missing>Math.max(0,remaining))continue;
    // This is a bounded preference heuristic, not a calibrated probability.
    const factor=missing===0?1:missing===1?.7:missing===2?.35:.12;
    const expectedGain=p.reward*factor*likelihood*Math.min(1,Math.max(0,remaining-1)/Math.max(1,missing*8));
    if(!best||expectedGain>best.expectedGain)best={word:p.word,reward:p.reward,missing,missingLetters,expectedGain,games:p.games,bots:p.bots};
  }
  return best;
}
function residualRack(ctx:WordContext,words:readonly string[]):WordContext {
  const myRack=new Map(ctx.myRack);for(const word of words)for(const c of word)myRack.set(c,(myRack.get(c)??0)-1);return {...ctx,myRack};
}
export interface HerculeDecision extends Omit<ScoringDecision,'policyVersion'> {
  policyVersion:typeof HERCULE_POLICY_VERSION;
  originalBaseline:number;
  acquisitionExtra:number;
  reason:'productive-baseline'|'extra-own-value'|'public-word-preference';
  publicGoal?:PublicGoal;
  letterAffinity:number;
}
export function herculeOffenseBid(ctx:BidContext,config:HerculeConfig):HerculeDecision {
  const old=scoringBid(ctx,[],'hercule',0);
  const result:HerculeDecision={...old,policyVersion:HERCULE_POLICY_VERSION,originalBaseline:old.baseline,acquisitionExtra:0,reason:'productive-baseline',letterAffinity:0};
  if(config.mode==='none'||!Number.isFinite(ctx.myBalance)||ctx.myBalance<1||!/^[A-Z]$/.test(ctx.letter)||(ctx.remainingTiles??Infinity)<=0)return result;
  const state=residualRack(ctx,old.expectedPreAuctionWords??[]),goal=publicHerculeGoal(state),myRack=new Map(state.myRack);myRack.set(ctx.letter,(myRack.get(ctx.letter)??0)+1);
  const after=publicHerculeGoal({...state,myRack}),productive=Math.max(old.ownGain??0,old.ownPotential)>=8;
  const affinity=HERCULE_LETTER_AFFINITY[ctx.letter]??0;
  const advances=goal?.missingLetters.includes(ctx.letter)&&goal.missing<=2;
  let desired=0;
  if(config.mode==='spending'&&productive)desired=Math.min(config.maxExtra,Math.max(0,Math.ceil(Math.max(old.ownGain??0,old.ownPotential)*1.1+2)-old.baseline));
  if(config.mode==='public'){
    const goalLift=advances&&goal?Math.max(0,(after?.expectedGain??0)-goal.expectedGain)*.75:0;
    const letterLift=productive?config.maxExtra*affinity*1.8:0;
    desired=Math.min(config.maxExtra,Math.ceil(Math.max(goalLift,letterLift)));
  }
  // Rare-letter priors never override duplicate, liquidity or endgame checks.
  if((state.myRack.get(ctx.letter)??0)>=2||((ctx.remainingTiles??Infinity)<=8&&(old.ownGain??0)===0))desired=0;
  const balance=Math.floor(ctx.myBalance),ending=(ctx.remainingTiles??Infinity)<=8||(ctx.progress??0)>=.88;
  const reserve=ending?0:Math.min(SCORING_SETTINGS.hercule.reserve,Math.floor(balance*.15));
  const cap=Math.max(0,Math.min(balance-reserve,Math.floor(balance*(ending?.95:.65))));
  result.amount=result.baseline=Math.max(old.baseline,Math.min(cap,old.baseline+desired));
  result.acquisitionExtra=result.amount-old.baseline;result.publicGoal=goal;result.letterAffinity=affinity;
  if(result.acquisitionExtra>0)result.reason=config.mode==='public'?'public-word-preference':'extra-own-value';
  return result;
}
export function chooseOffensiveHerculeWord(ctx:WordContext):string|null {return chooseScoringWord(ctx,'hercule');}
