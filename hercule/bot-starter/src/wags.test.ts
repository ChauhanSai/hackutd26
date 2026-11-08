import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, readFileSync, unlinkSync, rmdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chooseWord, decideBid } from "./strategy.js";
import { decideWagsBid, chooseWagsWord, assertWagsIdentity } from "./wags.js";
import { Study, selectProfile, type MatchResult } from "./study.js";

const rack = (letters: string) => {
  const result = new Map<string, number>();
  for (const letter of letters) result.set(letter,(result.get(letter)??0)+1);
  return result;
};

test("Wags bids substantially above ordinary hercule-v1 exploration", () => {
  for (const letter of ["A","E","X","Z"]) {
    const ctx = { letter,myBalance:100,myRack:rack(""),dictionary:["AT","AX"] };
    assert.ok(decideWagsBid(ctx) >= 3*decideBid(ctx));
  }
});
test("Wags pays an aggressive premium for a profitable completion", () => {
  const ctx = { myBalance:100,myRack:rack("QUI"),dictionary:["QUIZ"] };
  assert.ok(decideWagsBid({ ...ctx,letter:"Z" })>=33);
  assert.ok(decideWagsBid({ ...ctx,letter:"Z" }) > decideWagsBid({ ...ctx,letter:"X" }));
});
test("Wags respects its 60 percent budget and can still bid with little cash", () => {
  for (const myBalance of [0,1,4,10,11,15,100,100.5,-1,NaN]) {
    const amount = decideWagsBid({ letter:"Z",myBalance,myRack:rack("ZZZ"),dictionary:["ZZZZ"] });
    assert.ok(Number.isInteger(amount) && amount >= 0);
    if (Number.isFinite(myBalance) && myBalance >= 0) {
      assert.ok(amount <= myBalance);
      assert.ok(amount <= (myBalance < 1 ? 0 : Math.max(1,Math.floor(myBalance*0.6))));
      assert.ok(Math.floor(myBalance)-amount>=Math.min(6,Math.floor(Math.floor(myBalance)*0.15)));
    }
  }
  assert.equal(decideWagsBid({ letter:"A",myBalance:1,myRack:rack(""),dictionary:[] }),1);
});
test("Wags protects a large available payout from short-word fragmentation", () => {
  const ctx = { myRack:rack("TACTICSAX"),dictionary:["AX","ACTS","TIC","TACTICS"] };
  assert.equal(chooseWord(ctx),"TACTICS");
  assert.equal(chooseWagsWord(ctx),"TACTICS");
});
test("Wags consumes repeated letters correctly and notices mutations to the rack", () => {
  const ctx = { myRack:rack("AT"),dictionary:["TAT"] };
  assert.equal(chooseWagsWord(ctx),null);
  ctx.myRack.set("T",2);
  assert.equal(chooseWagsWord(ctx),"TAT");
});
test("Wags cashes an available long word", () => {
  assert.equal(chooseWagsWord({ myRack:rack("RETAINS"),dictionary:["RETAINS"] }),"RETAINS");
});
test("Wags cashes a profitable sequence without reusing letters", () => {
  const ctx = { myRack:rack("TACTICSAX"),dictionary:["AX","ACTS","TIC","TACTICS"] };
  const words: string[] = [];
  for (let i = 0; i < 10; i++) {
    const word = chooseWagsWord(ctx);
    if (!word) break;
    words.push(word);
    for (const letter of word) {
      assert.ok(ctx.myRack.get(letter)! > 0);
      ctx.myRack.set(letter,ctx.myRack.get(letter)!-1);
    }
  }
  assert.deepEqual(words,["TACTICS","AX"]);
  assert.equal([...ctx.myRack.values()].reduce((a,b)=>a+b,0),0);
});
test("Wags refuses hercule-v1's registered identity", () => {
  assert.doesNotThrow(()=>assertWagsIdentity("Wags"));
  assert.throws(()=>assertWagsIdentity("hercule-v1"),/separately registered bot named Wags/);
  assert.throws(()=>assertWagsIdentity(undefined),/Wags needs a token/);
});
test("Wags reports its own throughput without selecting a defensive profile", (t) => {
  const root = mkdtempSync(join(tmpdir(),"wags-report-"));
  const study = new Study(false,1,root,{ botName:"Wags",strategy:"wags" });
  t.after(()=>{
    for (const file of ["events.jsonl","results.json","REPORT.md"]) unlinkSync(join(study.dir,file));
    rmdirSync(study.dir); rmdirSync(root);
  });
  const assignment = study.assign("test",false,"wags");
  const result: MatchResult = { matchId:"test",...assignment,score:23,rank:1,opponents:[],
    defensiveBids:0,defensiveWins:0,defensiveSpend:0,totalSpend:10,
    words:[{word:"AX",reward:9},{word:"ACTS",reward:9},{word:"TIC",reward:5}] };
  study.complete(result);
  const report = readFileSync(join(study.dir,"REPORT.md"),"utf8");
  assert.ok(report.includes("# Wags results") && report.includes("Words of 2–4 letters"));
  assert.ok(!report.includes("hercule-v1") && !report.includes("Exploratory defensive choice"));
  assert.equal(JSON.parse(readFileSync(join(study.dir,"results.json"),"utf8")).selected,"wags");
  assert.equal(selectProfile([result]),"cautious");
});
