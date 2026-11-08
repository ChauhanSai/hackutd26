import assert from 'node:assert/strict';import {test} from 'node:test';
import {createLiveHerculePolicy,recoverHerculeKnowledge,HERCULE_LIVE_CONFIG,HERCULE_LIVE_VERSION} from './hercule-live.js';
import {makeMedianAgent} from './median-policy.js';import {HERCULE_LIVE_MODEL} from './hercule-median-settings.js';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';import {join,resolve,dirname,basename} from 'node:path';import {tmpdir} from 'node:os';import {Study} from './study.js';
const rack=(s:string)=>new Map([...new Set(s)].map(c=>[c,[...s].filter(x=>x===c).length]));
const ctx={myRack:rack('PERIE'),publicKnownRack:new Map<string,number>(),dictionary:['PERIQUE'],myBalance:100,myScore:0,remainingTiles:40,progress:.3,letter:'Q',auctionType:'Vickrey' as const,opponents:[]};
test('the active setup uses the frozen buying-only candidate without changing its decisions',()=>{
 assert.equal(HERCULE_LIVE_VERSION,'hercule-median-v5');assert.equal(HERCULE_LIVE_CONFIG.words,false);assert.equal(HERCULE_LIVE_CONFIG.valueScale,1.2);
 const a=createLiveHerculePolicy(),b=makeMedianAgent(HERCULE_LIVE_CONFIG,HERCULE_LIVE_MODEL);assert.equal(a.bid(ctx),b.bid(ctx));assert.equal(a.lastDecision?.baseline,a.lastDecision?.amount);
 a.reset?.();a.restore({initialRack:rack('PERIE'),seen:new Map(),round:0,lastWordRound:0,hiddenInitial:25});assert.equal(a.bid(ctx),b.lastDecision?.amount);
});
test('history restoration recovers our initial letters and all observed auctions after reconnect',()=>{
 const h={rack:new Map<string,number>(),ownId:1n,startingRackSize:5,participants:6,purchases:[{letter:'Q',winnerBotId:1n,at:1n},{letter:'U',winnerBotId:1n,at:2n},{letter:'Z',winnerBotId:2n,at:3n}],words:[{word:'PERIQUE',botId:1n,at:2n},{word:'QUIZ',botId:2n,at:3n}]};
 const k=recoverHerculeKnowledge(h)!;assert.deepEqual(k.initialRack,rack('PERIE'));assert.equal(k.seen.get('Z'),1);assert.equal(k.hiddenInitial,25);assert.equal(k.round,3);assert.equal(k.lastWordRound,2);
 assert.equal(recoverHerculeKnowledge({...h,words:[]}),undefined);
 const a=createLiveHerculePolicy(),b=createLiveHerculePolicy();a.restore(k);k.seen.set('Q',20);assert.equal(b.bid(ctx),createLiveHerculePolicy().bid(ctx));
});
test('reports distinguish the live candidate from older practice sessions',t=>{
 const root=mkdtempSync(join(tmpdir(),'hercule-v5-'));t.after(()=>{assert.equal(dirname(resolve(root)),resolve(tmpdir()));assert.ok(basename(root).startsWith('hercule-v5-'));rmSync(root,{recursive:true,force:true});});
 const s=new Study(false,2,root,{botName:'hercule-v1',strategy:'hercule',fixedProfile:'hercule',policyVersion:HERCULE_LIVE_VERSION});s.write();assert.equal(JSON.parse(readFileSync(join(s.dir,'results.json'),'utf8')).policyVersion,HERCULE_LIVE_VERSION);
});
