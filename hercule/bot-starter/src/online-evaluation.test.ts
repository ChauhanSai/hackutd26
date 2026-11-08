import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {join,dirname,resolve,basename} from 'node:path';
import {tmpdir} from 'node:os';
import {onlineEvaluation,assertBotIdentity} from './online-evaluation.js';
import {Study,type MatchResult} from './study.js';
test('online evaluation fixes the tested policy and validates limits and mode',()=>{
  assert.equal(onlineEvaluation({}),undefined);
  assert.deepEqual(onlineEvaluation({BOT_EVAL_MATCHES:'8'}),{matches:8,minutes:90,profile:'hercule',weight:0});
  assert.equal(onlineEvaluation({BOT_EVAL_MATCHES:'8',BOT_STRATEGY:'defensive'})?.weight,0);
  for(const env of [{BOT_EVAL_MATCHES:'0'},{BOT_EVAL_MATCHES:'8',BOT_STRATEGY:'victor'},{BOT_EVAL_MATCHES:'8',BOT_NAME:'Wags'},{BOT_EVAL_MATCHES:'8',BOT_TRAIN:'1'},{BOT_EVAL_MATCHES:'8',TOURNAMENT_ID:'1'},{BOT_EVAL_MATCHES:'8',BOT_EVAL_MINUTES:'NaN'}])assert.throws(()=>onlineEvaluation(env));
});
test('wrong credentials are rejected before any match entry',()=>{
  assert.doesNotThrow(()=>assertBotIdentity('hercule-v1','hercule-v1'));
  for(const name of ['Wags','VictorMateo',undefined])assert.throws(()=>assertBotIdentity('hercule-v1',name),/No matches entered/);
});
test('fixed evaluation never retunes and stops after full matches only',t=>{
  const root=mkdtempSync(join(tmpdir(),'fixed-online-'));t.after(()=>{assert.equal(dirname(resolve(root)),resolve(tmpdir()));assert.ok(basename(root).startsWith('fixed-online-'));rmSync(root,{recursive:true,force:true});});
  const study=new Study(false,2,root,{botName:'hercule-v1',strategy:'hercule',fixedProfile:'hercule'});
  assert.equal(study.assign('partial',true,'cautious').profile,'hercule');
  const game=(id:string,phase:MatchResult['phase']):MatchResult=>({matchId:id,profile:'hercule',phase,score:30,rank:2,opponents:[],defensiveBids:0,defensiveWins:0,defensiveSpend:0,totalSpend:20,words:[]});
  assert.equal(study.complete(game('partial','partial')),false);
  assert.equal(study.assign('1',false,'baseline').profile,'hercule');
  assert.equal(study.complete(game('1','normal')),false);
  const saved=JSON.parse(readFileSync(join(study.dir,'results.json'),'utf8'));assert.equal(saved.selected,'hercule');assert.equal(saved.weight,0);assert.equal(saved.fixedEvaluation,true);assert.equal(saved.plan,null);
  assert.equal(study.complete(game('2','normal')),true);assert.equal(study.complete(game('2','normal')),false);
});
