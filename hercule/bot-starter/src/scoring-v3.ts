// Wags/hercule-v1 only. VictorMateo keeps competition.ts + planning.ts v2.
import {dictionaryWordsByLength,chooseWord,completionValue,wordReward,LETTER_VALUE,type WordContext,type BidContext,type WordEntry} from "./strategy.js";
import {planRack,rackSize,type WordGoal} from "./planning.js";
import {contenderStrength,type Opponent} from "./intel.js";
import {LEADER_PRIORS} from "./leader-priors-v3.js";
import {type CompetitiveDecision,type PlayingStyle} from "./competition.js";

export const SCORING_POLICY_VERSION="scoring-pressure-v3";
export const SCORING_SETTINGS={
  wags:{common:13,other:9,rare:8,fraction:0.6,reserve:6,gain:1.0,cashOut:16,holdGain:3.5,denial:0.65,dualPremium:3},
  hercule:{common:12,other:9,rare:7,fraction:0.55,reserve:8,gain:0.9,cashOut:18,holdGain:3,denial:0.75,dualPremium:4},
} as const;
const FREQUENCY=[9,2,2,4,12,2,3,2,9,1,1,4,2,6,8,2,1,6,4,6,4,2,2,1,2,1];
interface Frontier {base:ReturnType<typeof planRack>;goal?:WordGoal;projected:number}
const caches=new WeakMap<readonly string[],Map<string,Frontier>>();
function addLetter(ctx:WordContext,letter:string): WordContext {
  const myRack=new Map(ctx.myRack);myRack.set(letter,(myRack.get(letter)??0)+1);return {...ctx,myRack};
}
function withoutWord(ctx:WordContext,word:string): WordContext {
  const myRack=new Map(ctx.myRack);for(const letter of word)myRack.set(letter,(myRack.get(letter)??0)-1);return {...ctx,myRack};
}
function spareScore(words:readonly string[],available:readonly number[]):number {
  const counts=[...available];let score=0;
  for(const word of words){const used=new Map<number,number>();for(const letter of word){const i=letter.charCodeAt(0)-65;used.set(i,(used.get(i)??0)+1);}
    if([...used].every(([i,n])=>counts[i]>=n)){for(const [i,n]of used)counts[i]-=n;score+=wordReward(word);}}
  return score;
}
// A separate goal frontier favors the multiplier ceiling while retaining v2's
// feasible whole-rack packing. This cache is not shared with VictorMateo.
export function scoringFrontier(ctx:WordContext): Frontier {
  const counts=Array.from({length:26},(_,i)=>ctx.myRack.get(String.fromCharCode(65+i))??0),key=counts.join(",");
  let cache=caches.get(ctx.dictionary);if(!cache){cache=new Map();caches.set(ctx.dictionary,cache);}
  const cached=cache.get(key);if(cached)return cached;
  const base=planRack(ctx),index=dictionaryWordsByLength(ctx.dictionary);
  const candidates:{entry:WordEntry;missing:number;likelihood:number;rank:number}[]=[];
  const seen=new Set<string>();
  let mask=0;for(let i=0;i<26;i++)if(counts[i])mask|=1<<i;
  for(let length=7;length<=Math.min(9,base.size+2);length++)for(const entry of index[length]??[]){
    const absent=entry.mask&~mask,first=absent&(absent-1);if((first&(first-1))!==0)continue;
    let missing=0,likelihood=1;
    for(let i=0;i<entry.letters.length;i+=2){const letter=entry.letters[i],needed=entry.letters[i+1],lack=Math.max(0,needed-counts[letter]);
      missing+=lack;if(missing>2||needed>FREQUENCY[letter]){likelihood=0;break;}likelihood*=Math.min(1,FREQUENCY[letter]/5)**lack;}
    if(!likelihood||missing<1)continue;
    const rank=entry.reward*(missing===1?0.65:0.3)*likelihood;
    if(candidates.length>=24&&rank<=candidates[candidates.length-1].rank)continue;
    const signature=entry.letters.join(",");if(seen.has(signature))continue;seen.add(signature);
    candidates.push({entry,missing,likelihood,rank});candidates.sort((a,b)=>b.rank-a.rank||(a.entry.word<b.entry.word?-1:1));
    if(candidates.length>24)seen.delete(candidates.pop()!.entry.letters.join(","));
  }
  let goal:WordGoal|undefined;
  for(const {entry,missing,likelihood}of candidates){const held=new Uint8Array(26),rest=[...counts];
    for(let i=0;i<entry.letters.length;i+=2){const letter=entry.letters[i],use=Math.min(rest[letter],entry.letters[i+1]);held[letter]=use;rest[letter]-=use;}
    const gain=entry.reward+spareScore(base.words,rest)-base.score,expectedGain=gain*(missing===1?0.65:0.3)*likelihood;
    if(gain>0&&(!goal||expectedGain>goal.expectedGain))goal={word:entry.word,missing,gain,expectedGain,held};
  }
  const result={base,goal,projected:base.score+(goal?.expectedGain??0)};
  if(cache.size>=1024)cache.delete(cache.keys().next().value!);cache.set(key,result);return result;
}
function frontierValue(frontier:Frontier,remaining:number) {
  return frontier.base.score+((frontier.goal?.missing??Infinity)<=Math.max(0,remaining-1)?frontier.goal?.expectedGain??0:0);
}
export function chooseScoringWord(ctx:WordContext,style:PlayingStyle): string|null {
  const settings=SCORING_SETTINGS[style],{base,goal}=scoringFrontier(ctx),word=base.words[0];if(!word)return null;
  const long=base.words.find(candidate=>candidate.length>=7);if(long)return long;
  // A competitive bid for a substantial immediate improvement can justify
  // waiting one auction. Do not cash away the completion's ingredients first.
  if(ctx.auctionLetter&&/^[A-Z]$/.test(ctx.auctionLetter)&&(ctx.remainingTiles??Infinity)>0){
    const gain=scoringFrontier(addLetter(ctx,ctx.auctionLetter)).base.score-base.score;
    if(gain>=8&&(ctx.committedBid??0)>=Math.min(9,Math.ceil(gain*0.5)))return null;
  }
  const flush=ctx.myBalance===undefined||ctx.myBalance<=settings.cashOut||(ctx.remainingTiles??Infinity)<=5||
    (ctx.progress??0)>=0.92||base.size>=13;
  if(flush)return word;
  if(goal&&goal.expectedGain>=Math.max(settings.holdGain,base.score*0.12)&&
    (ctx.remainingTiles??Infinity)>8&&(ctx.progress??0)<0.86){
    for(const candidate of base.words){const available=Array.from({length:26},(_,i)=>(ctx.myRack.get(String.fromCharCode(65+i))??0)-goal.held[i]);
      let fits=true;for(const letter of candidate)if(--available[letter.charCodeAt(0)-65]<0){fits=false;break;}if(fits)return candidate;}
    return null;
  }
  if(word.length>=5||wordReward(word)>=8&&wordReward(word)/word.length>=3.5||base.size>=10)return word;
  return null;
}
function strength(opponent:Opponent) {
  const prior=LEADER_PRIORS[opponent.name];
  return contenderStrength({...opponent,matchesPlayed:opponent.matchesPlayed??prior?.matchesPlayed,
    wins:opponent.wins??prior?.wins,rating:opponent.rating??prior?.rating});
}
export interface ScoringDecision extends CompetitiveDecision {
  ownPotential:number;
  leaderBias:number;
  policyVersion:typeof SCORING_POLICY_VERSION;
  expectedPreAuctionWord?:string;
  expectedPreAuctionWords?:string[];
}
export function scoringBid(ctx:BidContext,opponents:Opponent[],style:PlayingStyle,weight=1): ScoringDecision {
  const result:ScoringDecision={amount:0,baseline:0,threat:0,weight,ownGain:0,ownPotential:0,leaderBias:0,policyVersion:SCORING_POLICY_VERSION};
  if(!Number.isFinite(ctx.myBalance)||ctx.myBalance<1||!/^[A-Z]$/.test(ctx.letter)||(ctx.remainingTiles??Infinity)<=0)return result;
  const settings=SCORING_SETTINGS[style],balance=Math.floor(ctx.myBalance),remaining=ctx.remainingTiles??Infinity;
  const ending=remaining<=8||(ctx.progress??0)>=0.88;
  const reserve=ending?0:Math.min(settings.reserve,Math.floor(balance*0.15));
  const budget=Math.max(0,Math.min(balance-reserve,Math.max(1,Math.floor(balance*(ending?0.95:settings.fraction)))));
  function ownValue(rack:WordContext){const before=scoringFrontier(rack),after=scoringFrontier(addLetter(rack,ctx.letter));
    return {gain:Math.max(0,after.base.score-before.base.score),potential:Math.max(0,frontierValue(after,remaining)-frontierValue(before,remaining))};}
  let value=ownValue(ctx);
  function ownDesire(state:WordContext){const have=state.myRack.get(ctx.letter)??0,size=rackSize(state.myRack),common="AEIOUNRSTL".includes(ctx.letter);
    let explore:number=common?settings.common:(LETTER_VALUE[ctx.letter]??0)>=8?settings.rare:settings.other;
    if(have>=2)explore=Math.floor(explore*0.25);
    const vowels=[...state.myRack].reduce((sum,[letter,n])=>sum+("AEIOU".includes(letter)?n:0),0);
    if(size>=4&&(vowels/size<0.25&&"AEIOU".includes(ctx.letter)||vowels/size>0.55&&!"AEIOU".includes(ctx.letter)))explore+=2;
    if(size>=9&&value.gain===0&&value.potential<2)explore=Math.min(explore,2);
    if(ending&&value.gain===0)explore=0;
    return Math.max(explore,Math.ceil(settings.gain*Math.max(value.gain,value.potential)+(value.gain>0?2:0)));
  }
  let baseline=Math.min(budget,ownDesire(ctx));
  let residual:WordContext=ctx,earned=0;
  const cashWords:string[]=[];
  for(let n=0;n<=Math.floor(rackSize(ctx.myRack)/2);n++){
    const word=chooseScoringWord({...residual,myBalance:ctx.myBalance+earned,auctionLetter:ctx.letter,committedBid:baseline},style);
    if(!word)break;cashWords.push(word);earned+=wordReward(word);residual=withoutWord(residual,word);
  }
  if(cashWords.length){value=ownValue(residual);baseline=Math.min(budget,ownDesire(residual));result.expectedPreAuctionWord=cashWords[0];result.expectedPreAuctionWords=cashWords;}
  result.amount=result.baseline=baseline;result.ownGain=value.gain;result.ownPotential=value.potential;
  if(!Number.isFinite(weight)||weight<=0)return result;
  const scale=Math.min(1.15,Math.max(0,style==="hercule"?weight/1.6:weight));
  const rows=opponents.filter(opponent=>!opponent.isSimulated).map(opponent=>{
    const known=rackSize(opponent.knownRack),total=opponent.rackSize??known,exact=opponent.rackSize!==undefined&&total===known;
    const context={myRack:opponent.knownRack,dictionary:ctx.dictionary},ready=chooseWord(context),readyReward=ready?wordReward(ready):0;
    const raw=completionValue(context,ctx.letter),confidence=exact?1:0.6,credible=raw>=(exact?8:12)?raw*confidence:0;
    const quality=strength(opponent);
    const forecast=opponent.score+Math.max(readyReward+credible,total*2.5*quality);
    const canFund=opponent.balance>=1||readyReward>0||total>known;
    return {opponent,known,total,exact,raw,credible,quality,forecast,canFund};
  }).filter(row=>row.canFund).sort((a,b)=>b.forecast-a.forecast||b.quality-a.quality||a.opponent.id.localeCompare(b.opponent.id)).slice(0,2);
  let best=baseline;
  for(const row of rows){const useful=value.gain>=2||value.potential>=3;
    const priority=1+Math.min(0.2,Math.max(0,row.quality-1)*0.3);
    const dual=useful?Math.min(settings.dualPremium,Math.ceil(row.credible*0.1*scale)):0;
    const uncertain=useful&&!row.exact&&row.total>=6&&row.quality>1.1&&remaining>8?1:0;
    const pure=row.credible*settings.denial*priority*scale;
    const want=Math.max(baseline+dual+uncertain,pure);
    const extraCap=Math.min(Math.floor(balance*(style==="wags"?0.15:0.2)),Math.ceil(row.credible*0.5)+uncertain);
    const extra=Math.min(extraCap,Math.max(0,Math.floor(want)-baseline));
    const amount=Math.min(budget,baseline+Math.floor(extra*(ctx.auctionType==="FirstPrice"?0.7:1)));
    if(amount>best){best=amount;result.target=row.opponent.id;result.targetName=row.opponent.name;result.threat=row.raw;
      result.confidence=row.credible>0?(row.exact?"exact":"partial"):"uncertain";result.leaderBias=amount-baseline;}
  }
  result.amount=best;return result;
}
