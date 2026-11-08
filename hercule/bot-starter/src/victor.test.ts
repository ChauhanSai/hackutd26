import assert from "node:assert/strict";
import {test} from "node:test";
import {mkdtempSync,readFileSync,unlinkSync,rmdirSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {VictorMateoPolicy,assertVictorIdentity,VICTOR_POLICY_VERSION} from "./victor.js";
import {makeVictorAgent,simulateMatch,type ArenaAgent} from "./local-arena.js";
import {Study,selectProfile,type MatchResult} from "./study.js";
import {loadDictionary} from "./dictionary.js";
import {prepareDictionary} from "./strategy.js";

const rack=(text:string)=>{const r=new Map<string,number>();for(const c of text)r.set(c,(r.get(c)??0)+1);return r;};
const rival={id:"2",name:"Nijello",score:20,balance:60,isSimulated:false,knownRack:rack("QUI"),rackSize:3};
const ctx={letter:"E",myRack:rack("AT"),dictionary:["AT","QUIZ"],myBalance:100,myScore:0,
  opponents:[rival],remainingTiles:30,progress:0.2};

test("Victor opens aggressively and never treats starting cash as earned success",()=>{
  for(const myBalance of [100,200]){
    const policy=new VictorMateoPolicy(),decision=policy.bid({...ctx,myBalance});
    assert.equal(decision.mode,"build");assert.equal(decision.baseline,18);
  }
});
test("funded earned points trigger defense and reduce unproductive exploration",()=>{
  const policy=new VictorMateoPolicy();
  const decision=policy.bid({...ctx,myBalance:120,myScore:40});
  assert.equal(decision.mode,"defend");assert.equal(decision.baseline,3);
  assert.equal(policy.snapshot.transitions,1);
  assert.ok(decision.amount<=Math.floor(120*0.45));
});
test("relative cash advantage can trigger defense without an arbitrary clock cutoff",()=>{
  const policy=new VictorMateoPolicy();
  assert.equal(policy.bid({...ctx,myBalance:90,myScore:40,progress:0.1}).mode,"defend");
  const other=new VictorMateoPolicy();
  assert.equal(other.bid({...ctx,myBalance:90,myScore:40,progress:0.9,opponents:[{...rival,balance:100}]}).mode,"build");
});
test("a visible lead cannot hide a large unplayed rival rack",()=>{
  const policy=new VictorMateoPolicy();
  const decision=policy.bid({...ctx,myBalance:150,myScore:40,opponents:[{...rival,score:0,knownRack:rack(""),rackSize:20}]});
  assert.equal(decision.mode,"build");
  assert.ok(decision.position.rivalProjected>decision.position.ownProjected+12);
});
test("cash and score hysteresis prevent repeated build/defend flips",()=>{
  const policy=new VictorMateoPolicy(),base={...ctx,myScore:40,opponents:[{...rival,balance:100}]};
  assert.equal(policy.bid({...base,myBalance:120}).mode,"defend");
  assert.equal(policy.bid({...base,myBalance:90}).mode,"defend");
  assert.equal(policy.bid({...base,myBalance:54}).mode,"rebuild");
  assert.equal(policy.bid({...base,myBalance:70}).mode,"rebuild");
  assert.equal(policy.bid({...base,myBalance:120}).mode,"defend");
  assert.equal(policy.snapshot.transitions,3);
});
test("falling behind projected opposition returns to offense even with plenty of cash",()=>{
  const policy=new VictorMateoPolicy();
  policy.bid({...ctx,myBalance:150,myScore:40});
  assert.equal(policy.bid({...ctx,myBalance:150,myScore:40,opponents:[{...rival,score:120}]}).mode,"rebuild");
});
test("defense still blocks credible completions and buys our profitable completions",()=>{
  const policy=new VictorMateoPolicy();
  const block=policy.bid({...ctx,myBalance:120,myScore:40,letter:"Z"});
  assert.equal(block.mode,"defend");assert.equal(block.target,rival.id);
  assert.equal(block.confidence,"exact");assert.ok(block.amount>block.baseline);
  assert.ok(120-block.amount>=30);
  const own=policy.bid({...ctx,myBalance:120,myScore:40,myRack:rack("QUI"),letter:"Z",opponents:[{...rival,knownRack:rack(""),rackSize:0}]});
  assert.ok(own.baseline>=29);assert.equal(own.ownGain,33);
});
test("uncertain defense is capped and never invents hidden word completions",()=>{
  const decision=new VictorMateoPolicy().bid({...ctx,myBalance:120,myScore:40,opponents:[{...rival,knownRack:rack(""),rackSize:5}]});
  assert.equal(decision.mode,"defend");assert.equal(decision.threat,0);
  assert.equal(decision.confidence,"uncertain");assert.ok(decision.amount<=decision.baseline+3);
});
test("defense continues scoring and low cash funds rebuilding",()=>{
  const policy=new VictorMateoPolicy();
  policy.bid({...ctx,myBalance:120,myScore:40});
  assert.equal(policy.word({...ctx,myBalance:120,myScore:40,myRack:rack("RETAINS"),dictionary:["RETAINS"]}),"RETAINS");
  assert.equal(policy.word({...ctx,myBalance:20,myScore:40}),"AT");
  assert.equal(policy.snapshot.mode,"rebuild");
});
test("no real competitors means no reason to sit back in defense",()=>{
  const policy=new VictorMateoPolicy();
  assert.equal(policy.bid({...ctx,myBalance:200,myScore:100,opponents:[{...rival,isSimulated:true}]}).mode,"build");
});
test("budgets stay affordable across auction types and the final unusable tile is skipped",()=>{
  for(const auctionType of ["FirstPrice","Vickrey"] as const)for(const myBalance of [0,1,4,10,54,55,90,120,120.5,NaN,-1]){
    const policy=new VictorMateoPolicy();
    policy.bid({...ctx,myBalance:150,myScore:40});
    const decision=policy.bid({...ctx,myBalance,myScore:40,letter:"Z",auctionType});
    assert.ok(Number.isSafeInteger(decision.amount)&&decision.amount>=0);
    if(Number.isFinite(myBalance)&&myBalance>=0)assert.ok(decision.amount<=Math.floor(myBalance));
  }
  const policy=new VictorMateoPolicy();
  assert.equal(policy.bid({...ctx,myBalance:120,myScore:40,letter:"Z",remainingTiles:0}).amount,0);
  assert.equal(policy.word({...ctx,myBalance:120,myScore:40,remainingTiles:0}),"AT");
});
test("arena reuse resets the hybrid for each match and reproduces outcomes",()=>{
  const agent=makeVictorAgent();
  assert.equal(agent.bid({...ctx,myBalance:120,myScore:40,publicKnownRack:rack(""),auctionType:"Vickrey"}),3);
  const inert:ArenaAgent={name:"idle",kind:"fixture",bid:()=>0,word:()=>null};
  const options={seed:8,agents:[agent,inert],dictionary:["AT"],bag:["A","X","T"],startingRackSize:1,orderedBag:true};
  const first=simulateMatch(options);
  assert.deepEqual(simulateMatch(options),first);
  const opening={...ctx,myBalance:90,myScore:40,opponents:[{...rival,balance:100}],publicKnownRack:rack(""),auctionType:"Vickrey" as const};
  assert.equal(agent.bid(opening),makeVictorAgent().bid(opening));
  assert.ok(agent.bid(opening)>3);
});
test("reset and separate instances do not inherit another game's defensive phase",()=>{
  const policy=new VictorMateoPolicy();policy.bid({...ctx,myBalance:120,myScore:40});
  assert.equal(new VictorMateoPolicy().snapshot.mode,"build");
  policy.reset();assert.equal(policy.snapshot.mode,"build");assert.equal(policy.snapshot.transitions,0);
});
test("VictorMateo requires its own registered identity",()=>{
  assert.doesNotThrow(()=>assertVictorIdentity("VictorMateo"));
  for(const identity of ["hercule-v1","Wags",undefined])assert.throws(()=>assertVictorIdentity(identity),/separately registered bot named VictorMateo/);
});
test("hybrid reports are independent of defensive profile training",t=>{
  const root=mkdtempSync(join(tmpdir(),"victor-report-"));
  const study=new Study(false,1,root,{botName:"VictorMateo",strategy:"victor"});
  t.after(()=>{for(const file of ["events.jsonl","results.json","REPORT.md"])unlinkSync(join(study.dir,file));rmdirSync(study.dir);rmdirSync(root);});
  const assignment=study.assign("m1",false);
  const result:MatchResult={matchId:"m1",...assignment,score:42,rank:2,opponents:[],
    defensiveBids:2,defensiveWins:1,defensiveSpend:15,totalSpend:50,words:[{word:"RETAINS",reward:21}],hybrid:{mode:"rebuild",transitions:2}};
  study.complete(result);
  const saved=JSON.parse(readFileSync(join(study.dir,"results.json"),"utf8"));
  assert.equal(saved.selected,"victor");assert.equal(saved.policyVersion,VICTOR_POLICY_VERSION);assert.equal(saved.weight,null);
  const report=readFileSync(join(study.dir,"REPORT.md"),"utf8");
  assert.ok(report.includes("# VictorMateo results")&&report.includes("Mode changes"));
  assert.ok(!report.includes("hercule-v1")&&!report.includes("Exploratory defensive choice"));
  assert.equal(selectProfile([result]),"cautious");
  assert.throws(()=>new Study(true,1,root,{botName:"VictorMateo",strategy:"victor"}),/only for hercule-v1/);
});
test("full-dictionary hybrid decisions fit the auction window",t=>{
  const dictionary=loadDictionary();prepareDictionary(dictionary);
  const policy=new VictorMateoPolicy(),full={...ctx,dictionary,myRack:rack("RETAINAX"),opponents:[rival,{...rival,id:"3",name:"Chinese Dante",rackSize:8}]};
  const start=performance.now();policy.bid({...full,letter:"S"});
  policy.bid({...full,letter:"S",myBalance:150,myScore:100});
  const elapsed=performance.now()-start;t.diagnostic(`Two hybrid phase decisions: ${elapsed.toFixed(1)} ms`);
  assert.ok(elapsed<600);
});
