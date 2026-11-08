// Conservative experimental integration; production policies stay unchanged.
import {makeValueAgent,VALUE_CONTROL,type ValueConfig,type ValueDecision} from './value-agent.js';
import {planRack} from './planning.js';
import {priceFor,rackText,type ValueMarket,type ValueKnowledge} from './value-planning.js';
import {type ArenaAgent,type ArenaView} from './local-arena.js';
export interface OverlayConfig extends ValueConfig {overlay?:'premium'|'premium-hold'|'blind'}
export const OVERLAY_VARIANTS:OverlayConfig[]=[VALUE_CONTROL,
  {id:'premium-bids',goals:false,bids:true,defense:'current',overlay:'premium'},
  {id:'premium-joint',goals:false,bids:true,defense:'current',overlay:'premium-hold'},
  {id:'matched-extra',goals:false,bids:true,defense:'current',overlay:'blind'}];
export function makeOverlayAgent(config:OverlayConfig,market:ValueMarket,trace?:(e:Record<string,unknown>)=>void){
  if(!config.overlay)return makeValueAgent(VALUE_CONTROL,market,trace);
  const control=makeValueAgent(VALUE_CONTROL,market),proposal=makeValueAgent({id:'proposal',goals:false,bids:true,defense:'current'},market,trace?e=>trace({event:'planning-detail',proposal:e}):undefined);
  const diagnostics={goalChanges:0,wordHolds:0,pressureBids:0,planningCycles:0,premiumRaises:0,premiumCoins:0,completionHolds:0};
  const agent:ArenaAgent&{lastDecision?:ValueDecision;diagnostics:typeof diagnostics;restore:(k:ValueKnowledge)=>void}={name:'Candidate',kind:'hercule',diagnostics,
    restore(k){control.restore(k);proposal.restore(k);},
    reset(){control.reset?.();proposal.reset?.();agent.lastDecision=undefined;for(const k of Object.keys(diagnostics))diagnostics[k as keyof typeof diagnostics]=0;},
    bid(ctx){control.bid(ctx);proposal.bid(ctx);const old=control.lastDecision!,p=proposal.lastDecision!,balance=Math.max(0,Math.floor(ctx.myBalance)),ending=ctx.remainingTiles<=8||ctx.progress>=.88;
      const budget=Math.max(0,Math.min(Math.floor(balance*(ending?.95:.65)),balance-(ending?0:Math.min(8,Math.floor(balance*.1)))));
      let desired=old.baseline;
      if(config.overlay==='blind'){const worth=Math.max(old.ownGain,old.ownPotential);if(worth>=8)desired=Math.min(old.baseline+6,Math.ceil(worth*1.1+2));}
      else if(p.ownGain>=8||p.ownPotential>=6)desired=Math.min(old.baseline+6,Math.max(old.baseline,p.baseline));
      const baseline=Math.max(old.baseline,Math.min(budget,desired)),amount=ctx.remainingTiles<=0?0:Math.max(old.amount,Math.min(budget,baseline+Math.max(0,old.amount-old.baseline)));
      agent.lastDecision={...p,baseline:ctx.remainingTiles<=0?0:baseline,amount,reason:baseline>old.baseline?config.overlay==='blind'?'matched-extra-spending':'supported-premium-goal':'preserved-current-baseline'};
      if(baseline>old.baseline){diagnostics.premiumRaises++;diagnostics.premiumCoins+=baseline-old.baseline;}
      if(amount>baseline)diagnostics.pressureBids++;diagnostics.goalChanges=proposal.diagnostics.goalChanges;diagnostics.planningCycles=proposal.diagnostics.planningCycles;
      trace?.({event:'bid',rack:rackText(ctx.myRack),balance:ctx.myBalance,score:ctx.myScore,remaining:ctx.remainingTiles,letter:ctx.letter,originalBaseline:old.baseline,decision:agent.lastDecision});return amount;
    },
    word(ctx:ArenaView){const word=proposal.word(ctx);let hold=false;
      if(config.overlay==='premium-hold'&&word&&ctx.auctionLetter&&ctx.remainingTiles>0&&(ctx.committedBid??0)>=priceFor(market,ctx.auctionLetter,ctx.progress).top+1){
        const rack=new Map(ctx.myRack);rack.set(ctx.auctionLetter,(rack.get(ctx.auctionLetter)??0)+1);hold=planRack({...ctx,myRack:rack}).score-planRack(ctx).score>=18;}
      if(hold)diagnostics.completionHolds++;if(hold||!word)diagnostics.wordHolds++;
      trace?.({event:'word',rack:rackText(ctx.myRack),balance:ctx.myBalance,remaining:ctx.remainingTiles,word:hold?null:word,originalWord:word,reason:hold?'known-premium-completion':'preserved-v3-word-policy'});return hold?null:word;
    },
  };return agent;
}
