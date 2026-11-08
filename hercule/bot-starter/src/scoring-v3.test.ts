import assert from "node:assert/strict";
import {test} from "node:test";
import {scoringBid,chooseScoringWord,SCORING_SETTINGS} from "./scoring-v3.js";
import {VictorMateoPolicy} from "./victor.js";
import {loadDictionary} from "./dictionary.js";
import {prepareDictionary} from "./strategy.js";

const rack=(text:string)=>{const r=new Map<string,number>();for(const letter of text)r.set(letter,(r.get(letter)??0)+1);return r;};
const context={myRack:rack("QUI"),dictionary:["QI","QUIZ"],myBalance:20,myScore:0,remainingTiles:30,progress:0.2};
const leader={id:"2",name:"Nijello",score:80,balance:100,isSimulated:false,knownRack:rack("QUI"),rackSize:3};

test("a committed bid protects the completion instead of spending its letters first",()=>{
  for(const style of ["wags","hercule"] as const){
    const bid=scoringBid({...context,letter:"Z"},[],style,1.6);
    assert.ok(bid.amount>=9);assert.equal(bid.ownGain,22);
    assert.equal(chooseScoringWord({...context,auctionLetter:"Z",committedBid:bid.amount},style),null);
    assert.equal(chooseScoringWord({...context,myRack:rack("QUIZ")},style),"QUIZ");
  }
});
test("cash-first bids are valued after consuming words and never borrow unearned money",()=>{
  const bid=scoringBid({...context,myBalance:12,letter:"Z"},[],"wags");
  assert.deepEqual(bid.expectedPreAuctionWords,["QI"]);assert.equal(bid.ownGain,0);
  assert.ok(bid.amount<=Math.floor(12*SCORING_SETTINGS.wags.fraction));
  const many=scoringBid({...context,myRack:rack("TACTICSAX"),dictionary:["TACTICS","AX","ACTS","TIC"],letter:"Z",myBalance:100},[],"wags");
  assert.deepEqual(many.expectedPreAuctionWords,["TACTICS","AX"]);assert.equal(many.ownGain,0);
});
test("holding requires an actual commitment and a usable award",()=>{
  for(const style of ["wags","hercule"] as const){
    assert.equal(chooseScoringWord({...context,myBalance:12,auctionLetter:"Z",committedBid:0},style),"QI");
    assert.equal(chooseScoringWord({...context,auctionLetter:"Z",committedBid:20,remainingTiles:0},style),"QI");
    assert.equal(scoringBid({...context,letter:"Z",remainingTiles:0},[],style).amount,0);
  }
});
test("the scoring frontier preserves a strong seven-letter plan and cashes spare tiles",()=>{
  const ctx={...context,myRack:rack("RETAINAX"),dictionary:["AT","AX","RETAINS"],myBalance:100};
  for(const style of ["wags","hercule"] as const){
    assert.equal(chooseScoringWord(ctx,style),"AX");
    assert.equal(chooseScoringWord({...ctx,myRack:rack("RETAIN")},style),null);
    assert.equal(chooseScoringWord({...ctx,myRack:rack("RETAINS")},style),"RETAINS");
  }
});
test("a leader's completion gives a small premium to a letter useful to us too",()=>{
  const ctx={...context,myBalance:100,dictionary:["QUIZ"],letter:"Z"};
  for(const style of ["wags","hercule"] as const){
    const decision=scoringBid(ctx,[leader],style,style==="wags"?1:1.6);
    assert.equal(decision.target,leader.id);assert.equal(decision.confidence,"exact");
    assert.ok(decision.amount>decision.baseline);
    assert.ok(decision.amount-decision.baseline<=SCORING_SETTINGS[style].dualPremium);
  }
});
test("expensive attention is reserved for the leading threats, not a distant trailer",()=>{
  const ctx={...context,myBalance:100,dictionary:["QUIZ"],letter:"Z"};
  const opponents=[{...leader,score:0},
    {...leader,id:"3",name:"front",score:200,knownRack:rack(""),rackSize:0},
    {...leader,id:"4",name:"second",score:150,knownRack:rack(""),rackSize:0}];
  const result=scoringBid(ctx,opponents,"wags");assert.equal(result.amount,result.baseline);assert.equal(result.target,undefined);
});
test("uncertain leader pressure is tiny and never claims to know hidden letters",()=>{
  const decision=scoringBid({...context,myBalance:100,dictionary:["QUIZ"],letter:"Z"},
    [{...leader,knownRack:rack("A"),rackSize:6}],"wags");
  assert.equal(decision.threat,0);assert.equal(decision.confidence,"uncertain");
  assert.ok(decision.amount<=decision.baseline+1);
});
test("a broke exact rival with no playable word cannot fund an auction attack",()=>{
  const decision=scoringBid({...context,myRack:rack("AT"),dictionary:["AT","QUIZ"],letter:"Z",myBalance:100},
    [{...leader,balance:0}],"hercule",1.6);
  assert.equal(decision.amount,decision.baseline);assert.equal(decision.target,undefined);
});
test("VictorMateo retains its v2 word behavior despite new auction context fields",()=>{
  const ctx={...context,myBalance:12,auctionLetter:"Z",committedBid:10,opponents:[]};
  assert.equal(new VictorMateoPolicy().word(ctx),"QI");
  assert.equal(chooseScoringWord(ctx,"wags"),null);
});
test("both auction types keep bids integral and affordable across balance limits",()=>{
  for(const style of ["wags","hercule"] as const)for(const auctionType of ["Vickrey","FirstPrice"] as const)
    for(const myBalance of [0,1,4,12,20,100,100.5,NaN,-1]){
      const result=scoringBid({...context,letter:"Z",myBalance,auctionType},[leader],style,1.6);
      assert.ok(Number.isSafeInteger(result.amount)&&result.amount>=0);
      if(Number.isFinite(myBalance)&&myBalance>=0)assert.ok(result.amount<=Math.floor(myBalance));
    }
});
test("full-dictionary scoring decisions fit the auction deadline",t=>{
  const dictionary=loadDictionary();prepareDictionary(dictionary);
  const opponents=[leader,{...leader,id:"3",name:"Chinese Dante",rackSize:8},
    {...leader,id:"4",name:"Metheny",knownRack:rack("RETAIN"),rackSize:8}];
  const start=performance.now();
  const result=scoringBid({...context,dictionary,myRack:rack("AEINORSTCLUMPDB"),letter:"H",myBalance:100},opponents,"wags");
  const elapsed=performance.now()-start;t.diagnostic(`v3 crowded-rack bid: ${elapsed.toFixed(1)} ms`);
  assert.ok(elapsed<700);assert.ok(result.amount<=100);
});
