import { dictionaryWordsByLength, LETTER_VALUE, wordReward, type WordContext, type WordEntry } from "./strategy.js";

export interface WordGoal {
  word: string;
  missing: number;
  gain: number;
  expectedGain: number;
  held: Uint8Array;
}
export interface RackPlan {
  score: number;
  words: string[];
  size: number;
  goal?: WordGoal;
  projectedScore: number;
  searchLimited: boolean;
}
const caches = new WeakMap<readonly string[], Map<string, RackPlan>>();
const SEARCH_LIMIT = 80;
const GOAL_LIMIT = 10;
const FREQUENCY=[9,2,2,4,12,2,3,2,9,1,1,4,2,6,8,2,1,6,4,6,4,2,2,1,2,1];
const lettersIn = (rack: Map<string, number>) => Array.from({length:26},(_,i)=>rack.get(String.fromCharCode(65+i))??0);
export const rackSize = (rack: Map<string, number>) => [...rack.values()].reduce((sum,n)=>sum+n,0);
function fits(entry: WordEntry, counts: readonly number[]) {
  for (let i=0;i<entry.letters.length;i+=2) if (counts[entry.letters[i]]<entry.letters[i+1]) return false;
  return true;
}
function consume(entry: WordEntry, counts: readonly number[]) {
  const rest = [...counts];
  for (let i=0;i<entry.letters.length;i+=2) rest[entry.letters[i]]-=entry.letters[i+1];
  return rest;
}

// Exact when the deterministic search budget suffices; otherwise a feasible
// greedy remainder bounds work. Neither branch invents or reuses a tile.
export function planRack(ctx: WordContext): RackPlan {
  const counts=lettersIn(ctx.myRack), key=counts.join(",");
  let cache=caches.get(ctx.dictionary);
  if (!cache) { cache=new Map(); caches.set(ctx.dictionary,cache); }
  const prior=cache.get(key);
  if (prior) return prior;
  const size=counts.reduce((s,n)=>s+n,0);
  let mask=0;
  for (let i=0;i<26;i++) if (counts[i]) mask|=1<<i;
  const entries=dictionaryWordsByLength(ctx.dictionary);
  const playable=new Map<string,WordEntry>();
  const future: {entry:WordEntry;missing:number;rank:number;likelihood:number}[]=[];
  const futureSignatures=new Set<string>();
  for (let length=2;length<=Math.min(size+2,entries.length-1);length++) {
    for (const entry of entries[length]??[]) {
      const absent=entry.mask&~mask;
      const withoutFirst=absent&(absent-1);
      if ((withoutFirst&(withoutFirst-1))!==0) continue;
      let missing=0;
      for (let i=0;i<entry.letters.length;i+=2) {
        missing+=Math.max(0,entry.letters[i+1]-counts[entry.letters[i]]);
        if (missing>2) break;
      }
      if (missing===0) {
        const signature=entry.letters.join(",");
        const old=playable.get(signature);
        if (!old || entry.word<old.word) playable.set(signature,entry);
      } else if (missing<=2 && length>=6 && length<=9) {
        let likelihood=1;
        for (let i=0;i<entry.letters.length;i+=2) {
          const index=entry.letters[i], needed=entry.letters[i+1];
          if (needed>FREQUENCY[index]) {likelihood=0;break;}
          likelihood*=Math.min(1,FREQUENCY[index]/5)**Math.max(0,needed-counts[index]);
        }
        const rank=entry.reward*(missing===1?0.7:0.35)*likelihood;
        if (likelihood>0 && (future.length<64 || rank>future[future.length-1].rank)) {
          const signature=entry.letters.join(",");
          if (futureSignatures.has(signature)) continue;
          futureSignatures.add(signature);
          future.push({entry,missing,likelihood,rank});
          future.sort((a,b)=>b.rank-a.rank||(a.entry.word<b.entry.word?-1:1));
          if (future.length>64) futureSignatures.delete(future.pop()!.entry.letters.join(","));
        }
      }
    }
  }
  const allCandidates=[...playable.values()].sort((a,b)=>b.reward-a.reward||b.word.length-a.word.length||(a.word<b.word?-1:1));
  const perLength=new Map<number,number>();
  // Retain alternatives at every length instead of letting thousands of
  // similar long words consume an auction's entire decision window.
  const candidates=allCandidates.length<=256?allCandidates:allCandidates.filter((entry,index)=>{
    const count=perLength.get(entry.word.length)??0;perLength.set(entry.word.length,count+1);
    return index<24 || count<8;
  });
  const memo=new Map<string,{score:number;words:string[]}>();
  let visited=0, searchLimited=candidates.length<allCandidates.length;
  function greedy(available: readonly number[]) {
    let rest=[...available], score=0;
    const words:string[]=[];
    // Keep all legal outlets in the cheap remainder pass, even when the
    // branching pool was pruned. Otherwise spare short words disappear.
    for (const entry of allCandidates) while(fits(entry,rest)) {
      rest=consume(entry,rest);score+=entry.reward;words.push(entry.word);
    }
    return {score,words};
  }
  function pack(available: readonly number[]): {score:number;words:string[]} {
    const state=available.join(","), cached=memo.get(state);
    if (cached) return cached;
    if (visited>=SEARCH_LIMIT) { searchLimited=true; return greedy(available); }
    visited++;
    let best=greedy(available);
    const ceiling=available.reduce((s,n,i)=>s+n*(LETTER_VALUE[String.fromCharCode(65+i)]??0)*3,0);
    for (const entry of candidates) {
      if (best.score===ceiling) break;
      // Stop sibling expansion too: merely bounding recursion would still
      // execute an unbounded number of expensive fallback packings.
      if (visited>=SEARCH_LIMIT) {searchLimited=true;break;}
      if (!fits(entry,available)) continue;
      const tail=pack(consume(entry,available)), score=entry.reward+tail.score;
      if (score>best.score) best={score,words:[entry.word,...tail.words]};
    }
    memo.set(state,best);return best;
  }
  const packed=pack(counts);
  let goal: WordGoal|undefined;
  const signatures=new Set<string>();
  let considered=0;
  for (const {entry,missing,likelihood} of future) {
    const signature=entry.letters.join(",");
    if (signatures.has(signature)) continue;
    signatures.add(signature);
    const held=new Uint8Array(26), rest=[...counts];
    for (let i=0;i<entry.letters.length;i+=2) {
      const index=entry.letters[i], used=Math.min(rest[index],entry.letters[i+1]);
      held[index]=used;rest[index]-=used;
    }
    const gain=entry.reward+pack(rest).score-packed.score;
    const expectedGain=gain*(missing===1?0.65:0.3)*likelihood;
    if (gain>0 && (!goal || expectedGain>goal.expectedGain)) goal={word:entry.word,missing,gain,expectedGain,held};
    if (++considered>=GOAL_LIMIT) break;
  }
  const result: RackPlan={score:packed.score,words:packed.words,size,goal,projectedScore:packed.score+(goal?.expectedGain??0),searchLimited};
  if (cache.size>=1024) cache.delete(cache.keys().next().value!);
  cache.set(key,result);return result;
}

export function choosePlannedWord(ctx: WordContext, style: "wags"|"hercule"): string|null {
  const plan=planRack(ctx), word=plan.words[0];
  if (!word) return null;
  // Context-free callers request a rack liquidation, not an economic hold.
  const liquidate=ctx.myBalance===undefined || ctx.myBalance<=(style==="wags"?24:22)
    || (ctx.remainingTiles??Infinity)<=6 || (ctx.progress??0)>=0.9 || plan.size>=14;
  if (liquidate || word.length>=7) return word;
  const goal=plan.goal;
  if (goal && goal.expectedGain>=4) {
    // Cash a word only from spare letters while protecting the promising plan.
    for (const candidate of plan.words) {
      const available=lettersIn(ctx.myRack);
      for (let i=0;i<26;i++) available[i]-=goal.held[i];
      let valid=true;
      for (const letter of candidate) if (--available[letter.charCodeAt(0)-65]<0) {valid=false;break;}
      if (valid) return candidate;
    }
    return null;
  }
  // A valuable five/six-letter payout or a premium short outlet can fund more
  // pressure. Ordinary small words wait for a better rack or a cash trigger.
  const reward=wordReward(word);
  if (word.length>=5 || reward/Math.max(1,plan.size)>=3 || plan.size>=10) return word;
  return null;
}
