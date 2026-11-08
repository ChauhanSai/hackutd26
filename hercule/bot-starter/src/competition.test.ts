import assert from "node:assert/strict";
import {test} from "node:test";
import {competitiveBid} from "./competition.js";
import {chooseHerculeWord} from "./defense.js";
import {observeRacks,contenderStrength,type Opponent} from "./intel.js";
import {loadDictionary} from "./dictionary.js";
import {prepareDictionary} from "./strategy.js";

const rack=(text:string)=>{const r=new Map<string,number>();for(const letter of text)r.set(letter,(r.get(letter)??0)+1);return r;};
const ctx={myRack:rack("AT"),myBalance:100,dictionary:["AT","QUIZ"],letter:"Z",remainingTiles:30};
const rival:Opponent={id:"2",name:"Nijello",score:0,balance:100,knownRack:rack("QUI"),rackSize:3,isSimulated:false};
test("proven contenders are prioritized even before their visible score rises",()=>{
  const ordinary={...rival,id:"3",name:"new rival",score:10};
  const result=competitiveBid(ctx,[ordinary,rival],"hercule",1.6);
  assert.equal(result.target,rival.id);
  assert.equal(result.confidence,"exact");
  assert.ok(result.amount>result.baseline);
});
test("partial information is discounted and hidden letters are not fabricated",()=>{
  const exact=competitiveBid(ctx,[rival],"hercule",0.5);
  const partial=competitiveBid(ctx,[{...rival,rackSize:8}],"hercule",0.5);
  assert.ok(exact.amount>partial.amount);
  const unknown=competitiveBid({...ctx,letter:"A"},[{...rival,knownRack:rack(""),rackSize:5}],"hercule");
  assert.equal(unknown.threat,0);
  assert.equal(unknown.confidence,"uncertain");
  assert.ok(unknown.amount-unknown.baseline<=6);
});
test("public rack tracking detects an empty rack and stays exact after new purchases",()=>{
  const events=[
    {time:1n,kind:"purchase" as const,botId:"2",text:"NS"},
    {time:2n,kind:"word" as const,botId:"2",text:"RETAINS"},
    {time:3n,kind:"purchase" as const,botId:"2",text:"QUI"},
  ];
  const result=observeRacks(events,5,["2"]).get("2")!;
  assert.equal(result.rackSize,3);assert.equal(result.exact,true);
  assert.deepEqual([...result.knownRack].filter(([,n])=>n>0).sort(),[["I",1],["Q",1],["U",1]]);
  assert.equal(observeRacks([events[1]],5,["2"]).get("2")!.exact,false);
});
test("live public statistics supersede stale name-based priors",()=>{
  const low=contenderStrength({...rival,matchesPlayed:200,wins:0,rating:1000});
  const high=contenderStrength({...rival,name:"new champion",matchesPlayed:200,wins:100,rating:1800});
  assert.ok(high>low);
});
test("Wags buys harder; hercule-v1 retains more cash while both can block",()=>{
  const opening={...ctx,letter:"E",myRack:rack(""),dictionary:[]};
  assert.ok(competitiveBid(opening,[],"wags").amount>competitiveBid(opening,[],"hercule").amount);
  for(const style of ["wags","hercule"] as const){
    const block=competitiveBid(ctx,[rival],style,1.6);
    assert.ok(block.amount>block.baseline);
    assert.equal(competitiveBid({...ctx,remainingTiles:0},[rival],style).amount,0);
    assert.equal(competitiveBid(ctx,[{...rival,isSimulated:true}],style).amount,block.baseline);
    for(const myBalance of [0,1,4,10,11,25,100,100.5,NaN,-1]){
      const amount=competitiveBid({...ctx,myBalance},[rival],style).amount;
      assert.ok(Number.isSafeInteger(amount)&&amount>=0);
      if(Number.isFinite(myBalance)&&myBalance>=0)assert.ok(amount<=Math.floor(myBalance));
    }
  }
});
test("the last usable auction can fund a completion and the final window flushes words",()=>{
  assert.ok(competitiveBid({...ctx,myRack:rack("QUI"),remainingTiles:1},[],"hercule").amount>0);
  assert.equal(competitiveBid({...ctx,letter:"S",myRack:rack("RETAI"),dictionary:["RETAINS"],remainingTiles:1},[],"wags").amount,0);
  assert.equal(chooseHerculeWord({myRack:rack("AT"),dictionary:["AT"],myBalance:100,remainingTiles:0}),"AT");
});
test("a full-dictionary decision with five public opponents stays inside the auction budget",t=>{
  const dictionary=loadDictionary();prepareDictionary(dictionary);
  const opponents=["Nijello","Chinese Dante","Metheny","Lambi Lambiii","Doppio"].map((name,i)=>({...rival,name,id:String(i+1),knownRack:rack(["RETAIN","QUI","TACTIC","SILENT","CRANE"][i]),rackSize:8}));
  const start=performance.now();
  const decision=competitiveBid({...ctx,myRack:rack("RETAINAX"),dictionary,letter:"S"},opponents,"hercule",1.6);
  const elapsed=performance.now()-start;
  t.diagnostic(`Full-dictionary bid with five opponents: ${elapsed.toFixed(1)} ms`);
  assert.ok(elapsed<400);assert.ok(decision.amount>0);
  const crowded=performance.now();
  competitiveBid({...ctx,myRack:rack("AEINORSTCLUMPDB"),dictionary,letter:"H"},opponents,"wags",1);
  const crowdedElapsed=performance.now()-crowded;
  t.diagnostic(`Crowded-rack bid with five opponents: ${crowdedElapsed.toFixed(1)} ms`);
  assert.ok(crowdedElapsed<500);
});
