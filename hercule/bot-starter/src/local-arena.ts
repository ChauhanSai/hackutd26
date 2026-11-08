// Offline deterministic arena. No network, credentials, or leaderboard writes.
import { wordReward, chooseWord } from "./strategy.js";
import { defensiveBid, chooseHerculeWord, type Opponent } from "./defense.js";
import { decideWagsBid, chooseWagsWord } from "./wags.js";
import { createSampleBot, type PublicOpponentModel } from "./public-opponents.js";
import { VictorMateoPolicy } from "./victor.js";
import {decideHerculeBid,chooseHerculeWord as chooseCurrentHerculeWord} from './hercule.js';
const dictionarySets = new WeakMap<readonly string[],Set<string>>();

export interface ArenaView {
  myRack: Map<string, number>;
  publicKnownRack: Map<string, number>;
  myBalance: number;
  myScore: number;
  opponents: Opponent[];
  dictionary: readonly string[];
  progress: number;
  remainingTiles: number;
  auctionType: "Vickrey" | "FirstPrice";
  auctionLetter?:string;
  committedBid?:number;
}
export interface ArenaAgent {
  name: string;
  kind: "wags" | "hercule" | "victor" | "model" | "fixture";
  reset?:()=>void;
  bid: (ctx: ArenaView & { letter: string }) => number;
  word: (ctx: ArenaView) => string | null;
}
export interface PlayerResult {
  seat: number;
  name: string;
  kind: ArenaAgent["kind"];
  score: number;
  balance: number;
  spent: number;
  purchases: number;
  bidCount: number;
  bidTotal: number;
  words: number;
  wordLetters: number;
  shortWords: number;
  longWords: number;
  longScore: number;
  remainingRack: string;
  rank: number;
  sharedWin: boolean;
}
export interface ArenaResult {
  seed: number;
  auctionType: "Vickrey" | "FirstPrice";
  startingRackSize: number;
  auctions: number;
  unpurchasedTiles: number;
  players: PlayerResult[];
}

export function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value+0x6d2b79f5)>>>0;
    let mixed = Math.imul(value ^ value>>>15,value|1);
    mixed ^= mixed+Math.imul(mixed ^ mixed>>>7,mixed|61);
    return ((mixed ^ mixed>>>14)>>>0)/4294967296;
  };
}

function tieValue(seed: number, round: number, seat: number): number {
  let x = (seed ^ Math.imul(round+1,0x9e3779b9) ^ Math.imul(seat+1,0x85ebca6b))>>>0;
  x = Math.imul(x ^ x>>>16,0x7feb352d);
  x = Math.imul(x ^ x>>>15,0x846ca68b);
  return (x ^ x>>>16)>>>0;
}

export function settleAuction(bids: readonly number[], type: "Vickrey" | "FirstPrice", tieRanks: readonly number[]) {
  if (bids.some((bid)=>!Number.isSafeInteger(bid) || bid<0)) throw new Error("Bids must be nonnegative integers.");
  const sorted = bids.map((bid,seat)=>({bid,seat})).filter((row)=>row.bid>0)
    .sort((a,b)=>b.bid-a.bid || tieRanks[a.seat]-tieRanks[b.seat] || a.seat-b.seat);
  const winner = sorted[0];
  if (!winner) return { winner: undefined, paid:0, winningBid:0 };
  const paid = type === "FirstPrice" ? winner.bid : sorted[1]?.bid ?? Math.min(1,winner.bid);
  return { winner:winner.seat,paid,winningBid:winner.bid };
}

export function makeWagsAgent(): ArenaAgent {
  return { name:"Wags",kind:"wags",bid:decideWagsBid,word:chooseWagsWord };
}
// Explicit weights retain historical v3 comparisons. Normal local play uses v4.
export function makeHerculeAgent(weight?: number): ArenaAgent {
  if(weight===undefined)return {name:'hercule-v1',kind:'hercule',bid:ctx=>decideHerculeBid(ctx).amount,word:chooseCurrentHerculeWord};
  return { name:"hercule-v1",kind:"hercule",bid:(ctx)=>defensiveBid(ctx,ctx.opponents,weight).amount,word:chooseHerculeWord };
}
export function makeVictorAgent(): ArenaAgent {
  const policy=new VictorMateoPolicy();
  return {name:"VictorMateo",kind:"victor",bid:ctx=>policy.bid(ctx).amount,word:ctx=>policy.word(ctx),reset:()=>policy.reset()};
}
export function makeModelAgent(model: PublicOpponentModel): ArenaAgent {
  const sample = createSampleBot(model);
  return { name:model.name,kind:"model",bid:sample.decideBid,word:sample.chooseWord };
}

export function simulateMatch(options: {
  seed: number; agents: readonly ArenaAgent[]; dictionary: readonly string[]; bag: readonly string[];
  startingRackSize?: number; auctionType?: "Vickrey" | "FirstPrice"; orderedBag?: boolean;
}): ArenaResult {
  const { seed,agents,dictionary } = options;
  const startingRackSize = options.startingRackSize ?? 5;
  const auctionType = options.auctionType ?? "Vickrey";
  if (agents.length<2 || startingRackSize<0 || !Number.isInteger(startingRackSize) || agents.length*startingRackSize>=options.bag.length) throw new Error("Invalid local match configuration.");
  for(const agent of agents) agent.reset?.();
  const bag = [...options.bag];
  if (!options.orderedBag) {
    const random = seededRandom(seed);
    for (let i = bag.length-1; i>0; i--) { const j = Math.floor(random()*(i+1)); [bag[i],bag[j]]=[bag[j],bag[i]]; }
  }
  let validWords = dictionarySets.get(dictionary);
  if (!validWords) { validWords = new Set(dictionary); dictionarySets.set(dictionary,validWords); }
  const state = agents.map(()=>({ rack:new Map<string,number>(),known:new Map<string,number>(),balance:100,score:0,
    spent:0,purchases:0,bidCount:0,bidTotal:0,words:0,wordLetters:0,shortWords:0,longWords:0,longScore:0 }));
  let cursor = 0;
  const add = (rack: Map<string,number>,letter: string) => rack.set(letter,(rack.get(letter)??0)+1);
  for (const player of state) for (let n=0;n<startingRackSize;n++) add(player.rack,bag[cursor++]);
  const auctionCount = bag.length-cursor;
  let unpurchasedTiles = 0;

  function view(seat: number,round: number): ArenaView {
    const own = state[seat];
    return { myRack:new Map(own.rack),publicKnownRack:new Map(own.known),myBalance:own.balance,myScore:own.score,dictionary,
      opponents:state.flatMap((other,index)=>index===seat?[]:[{ id:String(index),name:agents[index].name,
        balance:other.balance,score:other.score,isSimulated:false,knownRack:new Map(other.known),rackSize:[...other.rack.values()].reduce((s,n)=>s+n,0) }]),
      progress:round/Math.max(1,auctionCount-1),remainingTiles:bag.length-cursor,auctionType };
  }
  for (let round=0;round<auctionCount;round++) {
    const letter = bag[cursor++];
    // Sealed opening bids: nobody sees opponents' bids or initial rack letters.
    const bids = agents.map((agent,seat)=>{
      const amount = agent.bid({ ...view(seat,round),letter });
      if (!Number.isSafeInteger(amount) || amount<0 || amount>state[seat].balance) throw new Error(`${agent.name} made an invalid bid.`);
      if (amount>0) { state[seat].bidCount++; state[seat].bidTotal+=amount; }
      return amount;
    });
    // During the auction, clients may cash existing words. We idealize RPC timing
    // and allow every policy to drain its chosen words; no strategy gets a
    // latency advantage. Submitted bids are not revised after these word plays.
    for (let seat=0;seat<agents.length;seat++) {
      const player = state[seat];
      for (let attempt=0;attempt<bag.length;attempt++) {
        const word = agents[seat].word({...view(seat,round),auctionLetter:letter,committedBid:bids[seat]});
        if (!word) break;
        if (!validWords.has(word) || !/^[A-Z]{2,}$/.test(word)) throw new Error(`${agents[seat].name} played an invalid word: ${word}`);
        const needed = new Map<string,number>();
        for (const char of word) add(needed,char);
        for (const [char,count] of needed) if ((player.rack.get(char)??0)<count) throw new Error(`${agents[seat].name} used unavailable tiles.`);
        for (const [char,count] of needed) {
          const remain = player.rack.get(char)!-count;
          if (remain) player.rack.set(char,remain); else player.rack.delete(char);
          player.known.set(char,Math.max(0,(player.known.get(char)??0)-count));
        }
        const reward = wordReward(word);
        player.score+=reward; player.balance+=reward; player.words++; player.wordLetters+=word.length;
        if (word.length<=4) player.shortWords++;
        if (word.length>=7) {player.longWords++;player.longScore+=reward;}
      }
    }
    const settled = settleAuction(bids,auctionType,agents.map((_agent,seat)=>tieValue(seed,round,seat)));
    if (settled.winner === undefined) unpurchasedTiles++;
    else {
      const winner = state[settled.winner];
      winner.balance-=settled.paid; winner.spent+=settled.paid; winner.purchases++;
      add(winner.rack,letter); add(winner.known,letter);
    }
    // The server ends immediately after the last auction. External bots cannot
    // submit a word using that final awarded letter after the match has ended.
  }
  const max = Math.max(...state.map((player)=>player.score));
  const players = state.map((player,seat): PlayerResult => ({ seat,name:agents[seat].name,kind:agents[seat].kind,
    score:player.score,balance:player.balance,spent:player.spent,purchases:player.purchases,bidCount:player.bidCount,bidTotal:player.bidTotal,
    words:player.words,wordLetters:player.wordLetters,shortWords:player.shortWords,longWords:player.longWords,longScore:player.longScore,
    remainingRack:[...player.rack].flatMap(([letter,count])=>Array<string>(count).fill(letter)).sort().join(""),
    rank:1+state.filter((other)=>other.score>player.score).length,sharedWin:player.score===max }));
  for (const player of players) if (player.balance<0 || player.balance!==100+player.score-player.spent) throw new Error("Cash conservation failed.");
  if (players.reduce((sum,player)=>sum+player.wordLetters+player.remainingRack.length,0)+unpurchasedTiles!==bag.length) throw new Error("Tile conservation failed.");
  return { seed,auctionType,startingRackSize,auctions:auctionCount,unpurchasedTiles,players };
}
