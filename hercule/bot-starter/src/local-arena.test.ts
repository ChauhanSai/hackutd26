import assert from "node:assert/strict";
import { test } from "node:test";
import { seededRandom,settleAuction,simulateMatch,type ArenaAgent } from "./local-arena.js";

test("neighboring seeds spread their first draw across the shuffle range",()=>{
  const bins = new Set(Array.from({length:128},(_,seed)=>Math.floor(seededRandom(seed)()*16)));
  assert.ok(bins.size>=12);
});

test("auction settlement matches second-price, first-price, reserve, and tie rules",()=>{
  assert.deepEqual(settleAuction([12,8,3],"Vickrey",[1,2,3]),{ winner:0,paid:8,winningBid:12 });
  assert.equal(settleAuction([12,8],"FirstPrice",[1,2]).paid,12);
  assert.equal(settleAuction([0,12],"Vickrey",[1,2]).paid,1);
  assert.equal(settleAuction([12,12],"Vickrey",[20,10]).winner,1);
  assert.equal(settleAuction([0,0],"Vickrey",[1,2]).winner,undefined);
});
const inert = (name: string): ArenaAgent => ({ name,kind:"fixture",bid:()=>0,word:()=>null });
test("views expose own dealt rack and conservative public counts, not opponents' dealt letters",()=>{
  let checked = false;
  const observer: ArenaAgent = { ...inert("observer"),bid:(ctx)=>{
    assert.equal([...ctx.myRack.values()].reduce((a,b)=>a+b,0),1);
    assert.equal(ctx.opponents[0].knownRack.size,0);
    assert.equal("myRack" in ctx.opponents[0],false);
    assert.equal("bids" in ctx,false);
    checked=true; return 0;
  } };
  simulateMatch({ seed:1,agents:[observer,inert("rival")],dictionary:[],bag:["A","X","T"],startingRackSize:1,orderedBag:true });
  assert.ok(checked);
});
test("final awarded tile cannot be cashed after the game ends",()=>{
  const target: ArenaAgent = { ...inert("target"),bid:()=>20,word:(ctx)=>
    ctx.myRack.has("A")&&ctx.myRack.has("T")?"AT":null };
  const result = simulateMatch({ seed:0,agents:[target,inert("rival")],dictionary:["AT"],bag:["A","X","T"],startingRackSize:1,orderedBag:true });
  assert.equal(result.players[0].score,0);
  assert.equal(result.players[0].remainingRack,"AT");
  assert.equal(result.players[0].balance,99);
});
test("repeat seeded matches reproduce outcomes and conserve cash and tiles",()=>{
  const target: ArenaAgent = { ...inert("target"),bid:()=>2,word:(ctx)=>
    (ctx.myRack.get("A")??0)>=2?"AA":null };
  const options = { seed:15,agents:[target,inert("rival")],dictionary:["AA"],bag:["A","A","A","T","X","A"],startingRackSize:1 };
  const a = simulateMatch(options), b = simulateMatch(options);
  assert.deepEqual(a,b);
  assert.equal(a.players[0].balance,100+a.players[0].score-a.players[0].spent);
});
test("invalid strategies fail instead of silently clipping bids or accepting illegal words",()=>{
  const bad: ArenaAgent = { ...inert("bad"),bid:()=>101 };
  assert.throws(()=>simulateMatch({ seed:0,agents:[bad,inert("rival")],dictionary:[],bag:["A","X","T"],startingRackSize:1 }),/invalid bid/);
});
