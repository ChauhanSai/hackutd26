import assert from "node:assert/strict";
import {test} from "node:test";
import {planRack,choosePlannedWord} from "./planning.js";
import {wordReward} from "./strategy.js";

const rack=(text:string)=>{const result=new Map<string,number>();for(const letter of text)result.set(letter,(result.get(letter)??0)+1);return result;};
test("packing sacrifices the largest individual word for a better whole-rack payout",()=>{
  const ctx={myRack:rack("QUIZPAX"),dictionary:["QUIZ","QUIP","ZAX"]};
  const plan=planRack(ctx);
  assert.equal(plan.score,41);
  assert.deepEqual(plan.words,["QUIP","ZAX"]);
  assert.equal(plan.searchLimited,false);
});
test("packing handles repeated copies and mutation without reusing tiles",()=>{
  const ctx={myRack:rack("AAT"),dictionary:["AT"]};
  assert.equal(planRack(ctx).score,2);
  ctx.myRack.set("T",2);
  assert.equal(planRack(ctx).score,4);
  assert.deepEqual(planRack(ctx).words,["AT","AT"]);
});
test("both styles retain a near-long-word plan while funded, but cash it when complete",()=>{
  const ctx={myRack:rack("RETAIN"),dictionary:["AT","RETAINS"],myBalance:100,remainingTiles:40,progress:0.2};
  for(const style of ["wags","hercule"] as const){
    assert.equal(choosePlannedWord(ctx,style),null);
    assert.equal(choosePlannedWord({...ctx,myBalance:5},style),"AT");
    assert.equal(choosePlannedWord({...ctx,remainingTiles:0},style),"AT");
    assert.equal(choosePlannedWord({...ctx,myRack:rack("RETAINS")},style),"RETAINS");
  }
});
test("spare letters can fund pressure without breaking a retained long plan",()=>{
  const ctx={myRack:rack("RETAINAX"),dictionary:["AT","AX","RETAINS"],myBalance:100,remainingTiles:40};
  assert.equal(choosePlannedWord(ctx,"wags"),"AX");
});
test("two-step goals do not depend on impossible repeated rare tiles",()=>{
  const ctx={myRack:rack("JAZ"),dictionary:["JAZZED","JAZZES"]};
  assert.equal(planRack(ctx).goal,undefined);
});
test("a bounded search still returns a feasible scored partition",()=>{
  const dictionary=["AA","AB","BB","AABB","AAAABB","AAAAAABB","AAAAAAAABB"];
  const counts=rack("AAAAAAAAAAAAAABBBBBBBBBB"), plan=planRack({myRack:counts,dictionary});
  for(const word of plan.words) for(const letter of word){assert.ok(counts.get(letter)!>0);counts.set(letter,counts.get(letter)!-1);}
  assert.equal(plan.score,plan.words.reduce((s,word)=>s+wordReward(word),0));
});
