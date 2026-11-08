// Experimental scoring planner. No online runner imports this module.
import {dictionaryWordsByLength,LETTER_VALUE,wordReward,type WordContext,type WordEntry} from './strategy.js';
import {planRack,rackSize} from './planning.js';
export const TILE_COUNTS=[9,2,2,4,12,2,3,2,9,1,1,4,2,6,8,2,1,6,4,6,4,2,2,1,2,1];
export interface PriceCell {paid:number;top:number}
export interface ValueMarket {letters:Record<string,PriceCell>;cells:Record<string,PriceCell>;fallback:PriceCell}
export interface ValueKnowledge {initialRack:Map<string,number>;seen:Map<string,number>;hiddenInitial:number;lastWordRound:number;round:number}
export interface ValueGoal {word:string;reward:number;missing:number;missingLetters:string[];held:number[];probability:number;cost:number;gain:number;utility:number;spareScore:number}
interface Near {entry:WordEntry;missing:number;missingLetters:string[];held:number[]}
interface Raw {base:ReturnType<typeof planRack>;near:Near[];alternatives:{word:string;reward:number}[]}
const caches=new WeakMap<readonly string[],Map<string,Raw>>();
export const rackText=(rack:Map<string,number>)=>[...rack].sort(([a],[b])=>a.localeCompare(b)).flatMap(([c,n])=>Array(Math.max(0,n)).fill(c)).join('');
export function priceFor(market:ValueMarket,letter:string,progress=0){return market.cells[`${letter}:${Math.min(2,Math.floor(progress*3))}`]??market.letters[letter]??market.fallback;}
export function rawValueFrontier(ctx:WordContext):Raw {
  const counts=Array.from({length:26},(_,i)=>ctx.myRack.get(String.fromCharCode(65+i))??0),key=counts.join(',');
  let cache=caches.get(ctx.dictionary);if(!cache){cache=new Map();caches.set(ctx.dictionary,cache);}const old=cache.get(key);if(old)return old;
  const base=planRack(ctx),index=dictionaryWordsByLength(ctx.dictionary),buckets:({entry:WordEntry;missing:number;rank:number;signature:string})[][]=[[],[],[],[]],common:typeof buckets=[[],[],[],[]],alternatives:{word:string;reward:number}[]=[];
  let mask=0;for(let i=0;i<26;i++)if(counts[i])mask|=1<<i;
  function keep(bucket:(typeof buckets)[number],entry:WordEntry,missing:number,rank:number){if(bucket.length>=24&&rank<=bucket.at(-1)!.rank)return;const signature=entry.letters.join(',');if(bucket.some(x=>x.signature===signature))return;
    bucket.push({entry,missing,rank,signature});bucket.sort((x,y)=>y.rank-x.rank||x.entry.word.localeCompare(y.entry.word));if(bucket.length>24)bucket.pop();}
  for(let length=2;length<=Math.min(10,base.size+3);length++)for(const entry of index[length]??[]){
    const absent=entry.mask&~mask,a=absent&(absent-1),b=a&(a-1);if((b&(b-1))!==0)continue;
    let missing=0,possible=true,frequency=1;for(let j=0;j<entry.letters.length;j+=2){const i=entry.letters[j],n=entry.letters[j+1];if(n>TILE_COUNTS[i]){possible=false;break;}const lack=Math.max(0,n-counts[i]);missing+=lack;frequency*=Math.min(1,TILE_COUNTS[i]/4)**lack;if(missing>3)break;}
    if(!possible||missing>3)continue;
    if(missing===0){if(alternatives.length<12||entry.reward>alternatives.at(-1)!.reward){alternatives.push({word:entry.word,reward:entry.reward});alternatives.sort((x,y)=>y.reward-x.reward||x.word.localeCompare(y.word));if(alternatives.length>12)alternatives.pop();}continue;}
    if(length<5)continue;
    const rank=entry.reward/(1+missing*0.3);keep(buckets[missing],entry,missing,rank);keep(common[missing],entry,missing,rank*frequency);
  }
  const unique=new Map([...buckets.flat(),...common.flat()].map(x=>[x.signature,x]));
  const near=[...unique.values()].map(({entry,missing})=>{const held=Array<number>(26).fill(0),missingLetters:string[]=[];
    for(let j=0;j<entry.letters.length;j+=2){const i=entry.letters[j],n=entry.letters[j+1];held[i]=Math.min(n,counts[i]);for(let k=counts[i];k<n;k++)missingLetters.push(String.fromCharCode(65+i));}return {entry,missing,held,missingLetters};});
  const result={base,near,alternatives};if(cache.size>=768)cache.delete(cache.keys().next().value!);cache.set(key,result);return result;
}
function combinationRatio(total:number,available:number,draws:number){let p=1;for(let i=0;i<draws;i++)p*=Math.max(0,total-available-i)/Math.max(1,total-i);return p;}
export function availability(total:number,available:number,draws:number,needed:number){
  if(needed===0)return 1;if(available<needed||draws<needed||total<needed)return 0;
  draws=Math.min(draws,total);let p=combinationRatio(total,available,draws),less=p;
  if(p===0){ // Small exact product for the rare fully covered populations.
    if(draws>total-available+needed-1)return 1;
    let sum=0;const choose=(n:number,k:number)=>{if(k<0||k>n)return 0;let v=1;for(let i=1;i<=Math.min(k,n-k);i++)v*= (n-i+1)/i;return v;};
    for(let k=0;k<needed;k++)sum+=choose(available,k)*choose(total-available,draws-k)/choose(total,draws);return Math.max(0,Math.min(1,1-sum));
  }
  for(let k=1;k<needed;k++){const denominator=(total-available-draws+k)*k;p=denominator>0?p*(available-k+1)*(draws-k+1)/denominator:0;less+=p;}
  return Math.max(0,Math.min(1,1-less));
}
function spare(words:readonly string[],counts:number[]){let score=0;for(const word of words){const need=Array<number>(26).fill(0);for(const c of word)need[c.charCodeAt(0)-65]++;
  if(need.every((n,i)=>n<=counts[i])){need.forEach((n,i)=>counts[i]-=n);score+=wordReward(word);}}return score;}
export function valueFrontier(ctx:WordContext,knowledge:ValueKnowledge,market:ValueMarket){
  const raw=rawValueFrontier(ctx),remaining=ctx.remainingTiles??68,total=remaining+knowledge.hiddenInitial,counts=Array.from({length:26},(_,i)=>ctx.myRack.get(String.fromCharCode(65+i))??0),goals:ValueGoal[]=[];
  for(const n of raw.near){if(n.missing>remaining)continue;const need=new Map<string,number>();for(const c of n.missingLetters)need.set(c,(need.get(c)??0)+1);
    let probability=1,cost=0;for(const [c,count] of need){const i=c.charCodeAt(0)-65,available=Math.max(0,TILE_COUNTS[i]-(knowledge.initialRack.get(c)??0)-(knowledge.seen.get(c)??0));
      probability*=availability(total,available,remaining,count)*0.8**count;cost+=priceFor(market,c,ctx.progress).paid*count;}
    if(!probability)continue;
    // Compute feasible spare payouts without reusing letters held for the goal.
    const rest=counts.map((count,i)=>count-n.held[i]),cashableSpare=spare(raw.base.words,rest),balance=ctx.myBalance??100;
    const reserve=Math.min(8,Math.floor(balance*0.1));if(cost>Math.max(0,balance-reserve)+cashableSpare)continue;
    const gain=n.entry.reward+cashableSpare-raw.base.score;
    const utility=probability*gain-0.35*cost*(0.5+0.5*probability)-0.4*n.missing;
    if(gain>0&&utility>0)goals.push({word:n.entry.word,reward:n.entry.reward,missing:n.missing,missingLetters:n.missingLetters,held:n.held,probability,cost,gain,utility,spareScore:cashableSpare});
  }
  goals.sort((a,b)=>b.utility-a.utility||b.reward-a.reward||a.word.localeCompare(b.word));
  return {...raw,goal:goals[0],goals:goals.slice(0,5),value:raw.base.score+(goals[0]?.utility??0)};
}
export function removeWord(rack:Map<string,number>,word:string){const rest=new Map(rack);for(const c of word){const n=(rest.get(c)??0)-1;if(n<0)throw new Error('Unavailable word letter');if(n)rest.set(c,n);else rest.delete(c);}return rest;}
export function valueWord(ctx:WordContext,knowledge:ValueKnowledge,market:ValueMarket){
  const frontier=valueFrontier(ctx,knowledge,market),word=frontier.base.words.slice().sort((a,b)=>wordReward(b)-wordReward(a)||a.localeCompare(b))[0];
  if(!word)return {word:null,reason:'no-playable-word',frontier};
  if(ctx.auctionLetter&&(ctx.remainingTiles??0)>0&&(ctx.committedBid??0)>0){const rack=new Map(ctx.myRack);rack.set(ctx.auctionLetter,(rack.get(ctx.auctionLetter)??0)+1);
    const gain=planRack({...ctx,myRack:rack}).score-frontier.base.score,threshold=Math.min(priceFor(market,ctx.auctionLetter,ctx.progress).top+1,Math.max(8,Math.ceil(gain*0.45)));
    if(gain>=12&&ctx.committedBid!>=threshold)return {word:null,reason:'committed-completion',frontier};}
  const ending=(ctx.remainingTiles??Infinity)<=5||(ctx.progress??0)>=0.92,lowCash=(ctx.myBalance??100)<=18,expired=knowledge.round-knowledge.lastWordRound>=10;
  if(ending||lowCash||expired||rackSize(ctx.myRack)>=14)return {word,reason:ending?'finish':lowCash?'fund-acquisition':expired?'hold-expired':'rack-limit',frontier};
  const goal=frontier.goal,worth=wordReward(word),ready=word.length>=7||worth>=30;
  if(ready&&(!goal||goal.missing>1||goal.utility<worth*0.4))return {word,reason:'valuable-ready-word',frontier};
  if(goal&&goal.utility>=Math.max(4,worth*0.2)&&(ctx.remainingTiles??0)>8){
    for(const candidate of frontier.base.words){const available=Array.from({length:26},(_,i)=>(ctx.myRack.get(String.fromCharCode(65+i))??0)-goal.held[i]);let fits=true;
      for(const c of candidate)if(--available[c.charCodeAt(0)-65]<0){fits=false;break;}if(fits)return {word:candidate,reason:'spare-income',frontier};}
    return {word:null,reason:'protect-valued-goal',frontier};
  }
  return {word:word.length>=5||worth>=10?word:null,reason:word.length>=5||worth>=10?'fund-next-goal':'wait-for-value',frontier};
}
