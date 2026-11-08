// Shared median buying policy; the live wrapper uses a frozen selected configuration.
import {herculeOffenseBid,chooseOffensiveHerculeWord} from './hercule-policy.js';
import {HERCULE_ACTIVE_CONFIG} from './hercule-settings.js';
import {rawValueFrontier,availability,TILE_COUNTS,priceFor,removeWord,rackText,type ValueMarket,type ValueKnowledge} from './value-planning.js';
import {planRack,rackSize} from './planning.js';
import {wordReward,type WordContext,type BidContext} from './strategy.js';
import {type ArenaView,type ArenaAgent} from './local-arena.js';

export interface MedianModel {median:{payout:number;cashOut:number};market:ValueMarket}
export interface MedianConfig {id:string;bids:boolean;words:boolean;bidScale:number;payoutOffset:number;cashOffset:number;valueScale:number}
export const MEDIAN_CENTER:MedianConfig={id:'median-center',bids:true,words:true,bidScale:1,payoutOffset:0,cashOffset:0,valueScale:1.15};
export const MEDIAN_VARIANTS:readonly MedianConfig[]=[
  {...MEDIAN_CENTER,id:'current-v4',bids:false,words:false},
  {...MEDIAN_CENTER,id:'median-bids',words:false},
  {...MEDIAN_CENTER,id:'median-words',bids:false},MEDIAN_CENTER,
  {...MEDIAN_CENTER,id:'bid-low',bidScale:.9},{...MEDIAN_CENTER,id:'bid-high',bidScale:1.1},
  {...MEDIAN_CENTER,id:'payout-low',payoutOffset:-3},{...MEDIAN_CENTER,id:'payout-high',payoutOffset:3},
  {...MEDIAN_CENTER,id:'cash-low',cashOffset:-3},{...MEDIAN_CENTER,id:'cash-high',cashOffset:3},
  {...MEDIAN_CENTER,id:'value-low',valueScale:1},{...MEDIAN_CENTER,id:'value-high',valueScale:1.3},
];
export interface MedianGoal {word:string;reward:number;missing:number;held:readonly number[];missingLetters:readonly string[];probability:number;cost:number;gain:number;spareScore:number;netReward:number;utility:number}
function spareScore(words:readonly string[],available:number[]){let score=0;for(const word of words){const need=Array<number>(26).fill(0);for(const c of word)need[c.charCodeAt(0)-65]++;if(need.every((n,i)=>n<=available[i])){need.forEach((n,i)=>available[i]-=n);score+=wordReward(word);}}return score;}
export function medianFrontier(ctx:WordContext,k:ValueKnowledge,model:MedianModel,target:number){
  const raw=rawValueFrontier(ctx),remaining=ctx.remainingTiles??68,total=remaining+k.hiddenInitial,goals:MedianGoal[]=[];
  const counts=Array.from({length:26},(_,i)=>ctx.myRack.get(String.fromCharCode(65+i))??0);
  for(const n of raw.near){if(n.entry.reward<target||n.missing>remaining)continue;
    const needs=new Map<string,number>();for(const c of n.missingLetters)needs.set(c,(needs.get(c)??0)+1);
    let probability=.9**n.missing,cost=0;
    for(const [c,count]of needs){const i=c.charCodeAt(0)-65,available=Math.max(0,TILE_COUNTS[i]-(k.initialRack.get(c)??0)-(k.seen.get(c)??0));probability*=availability(total,available,remaining,count);cost+=priceFor(model.market,c,ctx.progress).paid*count;}
    if(probability<=0)continue;
    // Do not reuse ingredients held for this goal in its spare-word income.
    const sparePayout=spareScore(raw.base.words,counts.map((count,i)=>count-n.held[i]));
    const gain=n.entry.reward+sparePayout-raw.base.score,netReward=gain-cost,utility=probability*gain-.12*cost;
    if(gain>0&&utility>0&&cost<=Math.max(0,ctx.myBalance??100)+sparePayout)goals.push({word:n.entry.word,reward:n.entry.reward,missing:n.missing,held:n.held,missingLetters:n.missingLetters,probability,cost,gain,spareScore:sparePayout,netReward,utility});
  }
  goals.sort((a,b)=>b.utility-a.utility||b.reward-a.reward||a.word.localeCompare(b.word));
  return {...raw,goals,goal:goals[0],value:raw.base.score+(goals[0]?.utility??0)};
}
export function chooseMedianWord(ctx:WordContext,k:ValueKnowledge,model:MedianModel,config:MedianConfig):string|null {
  if(!config.words)return chooseOffensiveHerculeWord(ctx);
  const target=model.median.payout+config.payoutOffset,cash=Math.max(2,model.median.cashOut+config.cashOffset),base=planRack(ctx),word=base.words[0];if(!word)return null;
  const best=base.words.slice().sort((a,b)=>wordReward(b)-wordReward(a)||a.localeCompare(b))[0];
  const ending=(ctx.remainingTiles??Infinity)<=5||(ctx.progress??0)>=.92;
  if(ending||(ctx.myBalance??100)<=cash||base.size>=14||k.round-k.lastWordRound>=12)return best;
  if(wordReward(best)>=target)return best;
  const f=medianFrontier(ctx,k,model,target);
  if(ctx.auctionLetter&&(ctx.remainingTiles??0)>0&&(ctx.committedBid??0)>0){
    const rack=new Map(ctx.myRack);rack.set(ctx.auctionLetter,(rack.get(ctx.auctionLetter)??0)+1);
    const after=planRack({...ctx,myRack:rack}),gain=after.score-f.base.score,price=priceFor(model.market,ctx.auctionLetter,ctx.progress);
    if(gain>=12&&after.words.some(w=>wordReward(w)>=target)&&(ctx.committedBid??0)>=Math.min(price.top+1,Math.max(10,Math.ceil(gain*.6))))return null;
  }
  const goal=f.goal;
  if(goal&&goal.utility>=Math.max(6,wordReward(best)*.25)&&goal.probability>=.35&&(ctx.remainingTiles??0)>10){
    for(const candidate of f.base.words){const available=Array.from({length:26},(_,i)=>(ctx.myRack.get(String.fromCharCode(65+i))??0)-goal.held[i]);let fits=true;for(const c of candidate)if(--available[c.charCodeAt(0)-65]<0){fits=false;break;}if(fits)return candidate;}
    return null;
  }
  return best.length>=5||wordReward(best)>=9?best:null;
}
export function makeMedianAgent(config:MedianConfig,model:MedianModel,trace?:(e:Record<string,unknown>)=>void){
  let k:ValueKnowledge,initialized=false;
  const diagnostics={marketRaises:0,premiumRaises:0,forecastCycles:0,forecastMismatches:0,wordHolds:0,goalWords:0};
  let expectedFirst:string|null|undefined,firstWordPending=false;
  const agent:ArenaAgent&{lastDecision?:Record<string,any>;diagnostics:typeof diagnostics;restore:(knowledge:ValueKnowledge)=>void}={name:'Candidate',kind:'hercule',diagnostics,
    restore(knowledge){k={...knowledge,initialRack:new Map(knowledge.initialRack),seen:new Map(knowledge.seen)};initialized=true;firstWordPending=false;},
    reset(){k={initialRack:new Map(),seen:new Map(),hiddenInitial:25,lastWordRound:0,round:0};initialized=false;agent.lastDecision=undefined;firstWordPending=false;for(const key of Object.keys(diagnostics))diagnostics[key as keyof typeof diagnostics]=0;},
    bid(ctx){if(!initialized){k.initialRack=new Map(ctx.myRack);initialized=true;}k.round++;k.seen.set(ctx.letter,(k.seen.get(ctx.letter)??0)+1);
      const old=herculeOffenseBid(ctx,HERCULE_ACTIVE_CONFIG);
      if(!config.bids){agent.lastDecision={...old,config:config.id,expectedWords:old.expectedPreAuctionWords??[]};firstWordPending=false;trace?.({event:'bid',rack:rackText(ctx.myRack),letter:ctx.letter,balance:ctx.myBalance,remaining:ctx.remainingTiles,decision:agent.lastDecision});return old.amount;}
      if(!Number.isFinite(ctx.myBalance)||ctx.myBalance<1||!/^[A-Z]$/.test(ctx.letter)||(ctx.remainingTiles??Infinity)<=0){agent.lastDecision={amount:0,baseline:0,weight:0,threat:0,expectedWords:[]};return 0;}
      const target=model.median.payout+config.payoutOffset,balance=Math.floor(ctx.myBalance),ending=(ctx.remainingTiles??Infinity)<=8||ctx.progress>=.88;
      const cap=Math.max(0,Math.min(balance-(ending?0:Math.min(8,Math.floor(balance*.12))),Math.floor(balance*(ending?.95:.65))));
      function proposal(commit:number){let state:WordContext=ctx,earned=0;const future={...k},words:string[]=[];
        for(let i=0;i<Math.floor(rackSize(ctx.myRack)/2);i++){const word=chooseMedianWord({...state,myBalance:balance+earned,auctionLetter:ctx.letter,committedBid:commit},future,model,config);if(!word)break;words.push(word);state={...state,myRack:removeWord(state.myRack,word)};earned+=wordReward(word);future.lastWordRound=k.round;}
        const before=medianFrontier(state,k,model,target),rack=new Map(state.myRack);rack.set(ctx.letter,(rack.get(ctx.letter)??0)+1);const after=medianFrontier({...state,myRack:rack},k,model,target);
        const gain=Math.max(0,after.base.score-before.base.score),potential=Math.max(0,after.value-before.value),price=priceFor(model.market,ctx.letter,ctx.progress),copies=state.myRack.get(ctx.letter)??0;
        const outlet=after.goals.find(g=>g.held[ctx.letter.charCodeAt(0)-65]>copies&&g.probability>=.35&&g.netReward>0&&g.cost<=Math.max(0,balance-price.paid)+g.spareScore);
        const valueCap=Math.ceil(config.valueScale*Math.max(gain,potential,outlet?.netReward??0)+2);
        const marketTarget=Math.ceil((ctx.auctionType==='FirstPrice'?price.paid*.85:price.top+1)*config.bidScale);
        let raised=0;if(copies<2&&(gain>=8||potential>=8||outlet))raised=Math.min(valueCap,marketTarget);
        if(ending&&gain===0)raised=0;
        const amount=Math.max(old.amount,Math.min(cap,raised));
        return {amount,baseline:amount,weight:0,threat:0,oldBaseline:old.amount,gain,potential,price,goal:outlet??after.goal,expectedWords:words,valueCap,marketTarget,config:config.id};
      }
      let d=proposal(old.amount),seen=new Set<number>([old.amount]);
      for(let i=0;i<4;i++){const next=proposal(d.amount);if(next.amount===d.amount){d=next;break;}if(seen.has(next.amount)){diagnostics.forecastCycles++;d=proposal(Math.min(d.amount,next.amount));break;}seen.add(d.amount);d=next;}
      agent.lastDecision=d;expectedFirst=d.expectedWords[0]??null;firstWordPending=true;
      if(d.amount>old.amount){diagnostics.marketRaises++;if('JQXZ'.includes(ctx.letter))diagnostics.premiumRaises++;}
      trace?.({event:'bid',rack:rackText(ctx.myRack),letter:ctx.letter,balance:ctx.myBalance,remaining:ctx.remainingTiles,decision:d});return d.amount;
    },
    word(ctx:ArenaView){const word=chooseMedianWord(ctx,k,model,config);if(firstWordPending){if((word??null)!==expectedFirst)diagnostics.forecastMismatches++;firstWordPending=false;}if(word){k.lastWordRound=k.round;if(wordReward(word)>=model.median.payout+config.payoutOffset)diagnostics.goalWords++;}else diagnostics.wordHolds++;
      trace?.({event:'word',rack:rackText(ctx.myRack),balance:ctx.myBalance,remaining:ctx.remainingTiles,word});return word;
    },
  };agent.reset!();return agent;
}
