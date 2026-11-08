import {scoringBid,chooseScoringWord,scoringFrontier} from './scoring-v3.js';
import {wordReward,type WordContext} from './strategy.js';
import {rackSize} from './planning.js';
import {valueFrontier,valueWord,rawValueFrontier,removeWord,rackText,priceFor,type ValueMarket,type ValueKnowledge} from './value-planning.js';
import {type ArenaView,type ArenaAgent} from './local-arena.js';
export interface ValueConfig {id:string;goals:boolean;bids:boolean;defense:'current'|'off'|'selective'}
export const VALUE_CONTROL:ValueConfig={id:'current-hercule',goals:false,bids:false,defense:'current'};
export const VALUE_VARIANTS:ValueConfig[]=[VALUE_CONTROL,{id:'goals-only',goals:true,bids:false,defense:'current'},{id:'bids-only',goals:false,bids:true,defense:'current'},{id:'joint',goals:true,bids:true,defense:'current'}];
export interface ValueDecision {amount:number;baseline:number;ownGain:number;ownPotential:number;reason:string;goal?:ReturnType<typeof valueFrontier>['goal'];predictedWords:string[];stable:boolean}
type Context=ArenaView&{letter:string};
export function makeValueAgent(config:ValueConfig,market:ValueMarket,trace?:(event:Record<string,unknown>)=>void):ArenaAgent&{lastDecision?:ValueDecision;diagnostics:Record<string,number>;restore:(k:ValueKnowledge)=>void} {
  let knowledge:ValueKnowledge={initialRack:new Map(),seen:new Map(),hiddenInitial:25,round:0,lastWordRound:0},lastProgress:number|undefined,lastGoal:string|undefined;
  const diagnostics={goalChanges:0,wordHolds:0,pressureBids:0,planningCycles:0};
  function choose(ctx:WordContext,k=knowledge){if(config.goals)return valueWord(ctx,k,market);const word=chooseScoringWord(ctx,'hercule');return {word,reason:word?'v3-cash':'v3-hold',frontier:undefined};}
  function own(ctx:Context){
    const before=valueFrontier(ctx,knowledge,market),rack=new Map(ctx.myRack);rack.set(ctx.letter,(rack.get(ctx.letter)??0)+1);
    const after=valueFrontier({...ctx,myRack:rack},knowledge,market),gain=Math.max(0,after.base.score-before.base.score),potential=Math.max(0,after.value-before.value);
    const balance=Math.floor(ctx.myBalance),ending=ctx.remainingTiles<=8||ctx.progress>=0.88,reserve=ending?0:Math.min(8,Math.floor(balance*0.1));
    const budget=Math.max(0,Math.min(balance-reserve,Math.floor(balance*(ending?0.95:0.65))));
    let explore='AEIOUNRSTL'.includes(ctx.letter)?8:6;
    if((ctx.myRack.get(ctx.letter)??0)>=2)explore=Math.floor(explore*0.25);
    if(rackSize(ctx.myRack)>=9&&gain<2&&potential<3)explore=Math.min(2,explore);
    if(ending&&gain===0)explore=0;
    const valueCap=Math.ceil(Math.max(gain,potential)*1.1+2);
    const price=priceFor(market,ctx.letter,ctx.progress),pursue=potential>=6||gain>=8;
    const priceBid=pursue&&price.top+1<=valueCap?Math.floor(price.top)+1:0;
    return {amount:ctx.remainingTiles<=0?0:Math.min(budget,Math.max(explore,Math.ceil(0.95*Math.max(gain,potential)+(gain>0?2:0)),priceBid)),gain,potential,frontier:before,budget};
  }
  function defend(ctx:Context,base:number,budget:number){
    if(config.defense==='off'||ctx.remainingTiles<=0)return base;
    const old=scoringBid(ctx,ctx.opponents,'hercule',1.6),extra=Math.max(0,old.amount-old.baseline);
    if(config.defense==='current')return Math.min(budget,base+extra);
    if(!extra||old.confidence==='uncertain'||old.threat<(old.confidence==='exact'?12:24))return base;
    const need=Math.floor(priceFor(market,ctx.letter,ctx.progress).top)+1;
    const valueCap=Math.ceil(old.threat*(old.confidence==='exact'?0.6:0.35));
    return need>base&&need<=Math.min(base+6,budget,valueCap)?need:base;
  }
  function forecastWords(ctx:Context,commit:number){let rack=new Map(ctx.myRack),earned=0;const words:string[]=[],futureKnowledge={...knowledge};
    for(let n=0;n<Math.floor(rackSize(ctx.myRack)/2);n++){const result=choose({...ctx,myRack:rack,myBalance:ctx.myBalance+earned,auctionLetter:ctx.letter,committedBid:commit},futureKnowledge);if(!result.word)break;
      words.push(result.word);rack=removeWord(rack,result.word);earned+=wordReward(result.word);futureKnowledge.lastWordRound=knowledge.round;}
    return {rack,earned,words};
  }
  const agent:ArenaAgent&{lastDecision?:ValueDecision;diagnostics:Record<string,number>;restore:(k:ValueKnowledge)=>void}={name:'Candidate',kind:'hercule',diagnostics,
    reset(){knowledge={initialRack:new Map(),seen:new Map(),hiddenInitial:25,round:0,lastWordRound:0};lastProgress=undefined;lastGoal=undefined;agent.lastDecision=undefined;for(const k of Object.keys(diagnostics))diagnostics[k as keyof typeof diagnostics]=0;},
    restore(k){knowledge={...k,initialRack:new Map(k.initialRack),seen:new Map(k.seen)};lastProgress=undefined;},
    bid(ctx){if(!Number.isFinite(ctx.myBalance)||ctx.myBalance<0)return 0;
      if(lastProgress!==ctx.progress){if(knowledge.round===0){knowledge.initialRack=new Map(ctx.myRack);knowledge.hiddenInitial=ctx.opponents.reduce((s,o)=>s+(o.rackSize??5),0);}knowledge.round++;knowledge.seen.set(ctx.letter,(knowledge.seen.get(ctx.letter)??0)+1);lastProgress=ctx.progress;}
      let decision:ValueDecision;
      if(!config.bids){const original=scoringBid(ctx,ctx.opponents,'hercule',1.6),ending=ctx.remainingTiles<=8||ctx.progress>=0.88,balance=Math.floor(ctx.myBalance),budget=Math.max(0,Math.min(balance-(ending?0:Math.min(8,Math.floor(balance*.15))),Math.max(1,Math.floor(balance*(ending?.95:.55)))));
        const amount=config.defense==='current'?original.amount:defend(ctx,original.baseline,budget);
        decision={amount,baseline:original.baseline,ownGain:original.ownGain??0,ownPotential:original.ownPotential,reason:'unchanged-v3-bid',predictedWords:original.expectedPreAuctionWords??[],stable:true};}
      else {let commit=own(ctx).amount,previous='',stable=false,words:string[]=[],current=own(ctx),amount=commit;
        for(let iteration=0;iteration<3;iteration++){const forecast=forecastWords(ctx,commit);words=forecast.words;
          current=own({...ctx,myRack:forecast.rack,myBalance:ctx.myBalance+forecast.earned});
          // Earnings forecast here inform future goals, never the opening cash cap.
          const ending=ctx.remainingTiles<=8||ctx.progress>=0.88,openingCap=Math.max(0,Math.min(Math.floor(ctx.myBalance)*(ending?0.95:0.65),Math.floor(ctx.myBalance)-(ending?0:Math.min(8,Math.floor(ctx.myBalance*0.1)))));
          const base=Math.min(Math.floor(openingCap),current.amount);amount=defend(ctx,base,Math.floor(openingCap));const signature=words.join('|');
          if(amount===commit&&signature===previous){stable=true;break;}previous=signature;commit=amount;
        }
        if(!stable)diagnostics.planningCycles++;
        const ending=ctx.remainingTiles<=8||ctx.progress>=0.88,openingCap=Math.max(0,Math.min(Math.floor(ctx.myBalance*(ending?0.95:0.65)),Math.floor(ctx.myBalance)-(ending?0:Math.min(8,Math.floor(ctx.myBalance*0.1)))));
        const baseline=Math.min(openingCap,current.amount);
        decision={amount:ctx.remainingTiles<=0?0:amount,baseline:ctx.remainingTiles<=0?0:baseline,ownGain:current.gain,ownPotential:current.potential,reason:'goal-marginal-value',goal:current.frontier.goal,predictedWords:words,stable};
      }
      if(decision.goal?.word!==lastGoal){if(lastGoal!==undefined)diagnostics.goalChanges++;lastGoal=decision.goal?.word;}
      if(decision.amount>decision.baseline)diagnostics.pressureBids++;agent.lastDecision=decision;
      if(trace){const raw=rawValueFrontier(ctx),frontier=config.bids||config.goals?valueFrontier(ctx,knowledge,market):undefined,old=scoringFrontier(ctx);
        trace({event:'bid',round:knowledge.round,rack:rackText(ctx.myRack),balance:ctx.myBalance,score:ctx.myScore,remaining:ctx.remainingTiles,letter:ctx.letter,decision,goal:frontier?.goal??old.goal,alternatives:raw.alternatives,consideredGoals:frontier?.goals});}
      return decision.amount;
    },
    word(ctx){const result=choose(ctx);if(result.word)knowledge.lastWordRound=knowledge.round;else if(rackSize(ctx.myRack)>0)diagnostics.wordHolds++;
      if(trace)trace({event:'word',round:knowledge.round,rack:rackText(ctx.myRack),balance:ctx.myBalance,score:ctx.myScore,remaining:ctx.remainingTiles,auctionLetter:ctx.auctionLetter,committedBid:ctx.committedBid,word:result.word,reason:result.reason,goal:result.frontier?.goal,alternatives:rawValueFrontier(ctx).alternatives,consideredGoals:result.frontier?.goals});return result.word;
    },
  };return agent;
}
