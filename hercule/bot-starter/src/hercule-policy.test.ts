import assert from 'node:assert/strict';
import {test} from 'node:test';
import {herculeOffenseBid,chooseOffensiveHerculeWord,publicHerculeGoal,HERCULE_OFFENSE_VARIANTS} from './hercule-policy.js';
import {scoringBid,chooseScoringWord} from './scoring-v3.js';
import {loadDictionary} from './dictionary.js';
import {prepareDictionary,wordReward} from './strategy.js';
const rack=(text:string)=>{const r=new Map<string,number>();for(const c of text)r.set(c,(r.get(c)??0)+1);return r;};
const ctx={myRack:rack('PERIUE'),dictionary:['PERIQUE'],myBalance:100,myScore:0,remainingTiles:30,progress:.3,letter:'Q'};
test('the offense-only control exactly matches productive v3 without pressure',()=>{
  for(const letter of ['A','Q','Z'])for(const myBalance of [0,1,12,100]){
    const c={...ctx,letter,myBalance},d=herculeOffenseBid(c,HERCULE_OFFENSE_VARIANTS[0]);
    assert.equal(d.amount,scoringBid(c,[],'hercule',0).amount);assert.equal(d.amount,d.baseline);assert.equal(d.threat,0);assert.equal(d.target,undefined);
  }
});
test('public preferences recognize supported legal word families without restricting the dictionary',()=>{
  const goal=publicHerculeGoal(ctx);assert.equal(goal?.word,'PERIQUE');assert.equal(goal?.reward,54);assert.deepEqual(goal?.missingLetters,['Q']);
  assert.equal(publicHerculeGoal({...ctx,dictionary:['AT']}),undefined);
  const d=herculeOffenseBid(ctx,HERCULE_OFFENSE_VARIANTS[3]);assert.ok(d.acquisitionExtra>0);assert.ok(d.acquisitionExtra<=6);
  assert.equal(chooseOffensiveHerculeWord({...ctx,myRack:rack('QUIZ'),dictionary:['QUIZ']}),'QUIZ');
});
test('rival identities, scores and racks cannot influence offensive purchases',()=>{
  const first={...ctx,opponents:[{name:'Nijello',score:1000,knownRack:rack('QUI')}]},second={...ctx,opponents:[]};
  for(const config of HERCULE_OFFENSE_VARIANTS)assert.deepEqual(herculeOffenseBid(first,config),herculeOffenseBid(second,config));
});
test('extra acquisition is bounded and affordable without borrowing expected word income',()=>{
  for(const config of HERCULE_OFFENSE_VARIANTS)for(const myBalance of [NaN,-1,0,1,4,12,20,100,100.5])for(const auctionType of ['Vickrey','FirstPrice'] as const){
    const d=herculeOffenseBid({...ctx,myBalance,auctionType},config);assert.ok(Number.isSafeInteger(d.amount)&&d.amount>=0);
    assert.ok(d.amount>=d.originalBaseline);assert.ok(d.acquisitionExtra<=config.maxExtra);assert.equal(d.amount,d.baseline);
    if(Number.isFinite(myBalance)&&myBalance>=0)assert.ok(d.amount<=Math.floor(myBalance));
  }
  for(const config of HERCULE_OFFENSE_VARIANTS){const d=herculeOffenseBid({...ctx,myRack:rack('QUI'),dictionary:['QI','QUIZ'],myBalance:12,letter:'Z'},config);assert.ok(d.amount<=12);}
});
test('duplicates and the unusable last auction cannot attract a public-data premium',()=>{
  for(const config of HERCULE_OFFENSE_VARIANTS){assert.equal(herculeOffenseBid({...ctx,remainingTiles:0},config).amount,0);
    assert.equal(herculeOffenseBid({...ctx,myRack:rack('QQ'),dictionary:['QI']},config).acquisitionExtra,0);}
});
test('word timing retains long payouts, liquidity and current-tile commitment handling',()=>{
  for(const text of ['QUI','QUIZ','TACTICSAX','RETAINAX'])for(const balance of [12,100]){
    const c={...ctx,myRack:rack(text),myBalance:balance,dictionary:['QI','QUIZ','TACTICS','AX','AT','RETAINS'],auctionLetter:'Z',committedBid:0};
    assert.equal(chooseOffensiveHerculeWord(c),chooseScoringWord(c,'hercule'));
  }
  assert.equal(wordReward('PERIQUE'),54);
});
test('full dictionary offensive decisions remain within the auction deadline',t=>{
  const dictionary=loadDictionary();prepareDictionary(dictionary);const start=performance.now();
  for(const config of HERCULE_OFFENSE_VARIANTS)herculeOffenseBid({...ctx,dictionary,myRack:rack('AEINORSTCLUMPDB'),letter:'H'},config);
  const elapsed=performance.now()-start;t.diagnostic(`four offensive variants: ${elapsed.toFixed(1)} ms`);assert.ok(elapsed<700);
});
