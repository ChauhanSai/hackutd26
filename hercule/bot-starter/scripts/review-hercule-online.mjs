import {readFileSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import assert from 'node:assert/strict';
const read=p=>JSON.parse(readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const campaign=read('runs/hercule-offense-online-active.json'),rows=[],events=[];
assert.equal(campaign.status,'stopped-by-user');
for(const directory of campaign.runDirectories){
  rows.push(...read(join(directory,'results.json')).results.map(g=>({...g,directory})));
  for(const line of readFileSync(join(directory,'events.jsonl'),'utf8').trim().split('\n'))try{events.push({...JSON.parse(line),directory});}catch{}
}
const unique=[...new Map(rows.map(g=>[g.matchId,g])).values()],full=unique.filter(g=>g.phase!=='partial');
const sum=a=>a.reduce((s,n)=>s+n,0),mean=a=>a.length?sum(a)/a.length:null;
function summary(games){
  const words=games.flatMap(g=>g.words),long=words.filter(w=>w.word.length>=7);
  return {games:games.length,wins:games.filter(g=>g.rank===1).length,score:mean(games.map(g=>g.score)),rank:mean(games.map(g=>g.rank)),winner:mean(games.map(g=>Math.max(g.score,...g.opponents.map(o=>o.score)))),spent:mean(games.map(g=>g.totalSpend)),cash:mean(games.map(g=>100+g.score-g.totalSpend)),longPerGame:long.length/games.length,longReward:mean(long.map(w=>w.reward)),longScoreShare:sum(long.map(w=>w.reward))/Math.max(1,sum(games.map(g=>g.score))),zeroScores:games.filter(g=>g.score===0).length};
}
const perGame=full.map(g=>{
  assert.equal(sum(g.words.map(w=>w.reward)),g.score);
  const ownId=String(events.find(e=>e.directory===g.directory&&e.event==='connected').data.botId);
  const d=events.filter(e=>e.event==='decision'&&String(e.data.matchId)===g.matchId);
  const start=events.find(e=>e.event==='match-start'&&String(e.data.matchId)===g.matchId),end=events.find(e=>e.event==='match-end'&&String(e.data.matchId)===g.matchId);
  const errors=events.filter(e=>e.event==='error'&&e.time>=start.time&&e.time<=end.time);
  const winner=[{id:ownId,name:'hercule-v1',score:g.score},...g.opponents].sort((a,b)=>b.score-a.score)[0];
  const winnerWords=winner.id===ownId?g.words:events.filter(e=>e.event==='word'&&String(e.data.matchId)===g.matchId&&String(e.data.botId)===winner.id).map(e=>({word:e.data.word,reward:Number(e.data.reward)}));
  const auctions=events.filter(e=>e.event==='auction-result'&&String(e.data.matchId)===g.matchId);
  const ownPurchases=auctions.filter(e=>String(e.data.winner)===ownId),winnerPurchases=auctions.filter(e=>String(e.data.winner)===winner.id);
  return {id:g.matchId,score:g.score,rank:g.rank,decisions:d.length,late:d.filter(e=>e.data.observedRemainingMs<=0).length,errors:errors.length,extras:d.filter(e=>e.data.acquisitionExtra>0).length,wordGoalBids:d.filter(e=>e.data.acquisitionExtra>0&&e.data.publicGoal?.missingLetters.includes(e.data.letter)).length,ownPurchases:ownPurchases.length,winnerPurchases:winnerPurchases.length,ownPremiumPurchases:ownPurchases.filter(e=>'JQXZ'.includes(e.data.letter)).length,winnerPremiumPurchases:winnerPurchases.filter(e=>'JQXZ'.includes(e.data.letter)).length,winner:winner.name,winnerScore:winner.score,winnerWordAudit:sum(winnerWords.map(w=>w.reward))===winner.score,winnerWords,defensiveBids:g.defensiveBids};
});
// A descriptive diagnostic slice, not a randomized or unbiased control group.
const completeIds=new Set(perGame.filter(g=>g.decisions===68&&g.late===0&&g.errors===0).map(g=>g.id));
const audited=perGame.filter(g=>g.winnerWordAudit),long=audited.flatMap(g=>g.winnerWords.filter(w=>w.word.length>=7));
const errors={};for(const e of events.filter(e=>e.event==='error')){const key=e.data.what+': '+e.data.message;errors[key]=(errors[key]??0)+1;}
const priors=read('experiments/hercule-offense/latest.json'),wordPriors=read(join(priors.directory,'word-priors.json'));
const common=new Set(wordPriors.words.map(w=>[...w.word].sort().join(''))),playedFamilies=full.flatMap(g=>g.words).filter(w=>common.has([...w.word].sort().join('')));
const result={generatedAt:new Date().toISOString(),overall:summary(full),excludedPartialGames:unique.length-full.length,completeOpeningLogSlice:summary(full.filter(g=>completeIds.has(g.matchId))),completeOpeningLogIds:[...completeIds],winnerWordAudits:audited.length,winnerLongPerGame:long.length/audited.length,winnerLongReward:mean(long.map(w=>w.reward)),winnerLongScoreShare:sum(long.map(w=>w.reward))/sum(audited.map(g=>g.winnerScore)),ownPurchasesPerGame:mean(perGame.map(g=>g.ownPurchases)),winnerPurchasesPerGame:mean(perGame.map(g=>g.winnerPurchases)),ownPremiumPurchases:sum(perGame.map(g=>g.ownPremiumPurchases)),winnerPremiumPurchases:sum(perGame.map(g=>g.winnerPremiumPurchases)),acquisitionRaises:sum(perGame.map(g=>g.extras)),wordGoalRaises:sum(perGame.map(g=>g.wordGoalBids)),observedPreferredFamiliesPlayed:playedFamilies,decisions:sum(perGame.map(g=>g.decisions)),lateLocalEstimates:sum(perGame.map(g=>g.late)),errors,defensiveBids:sum(perGame.map(g=>g.defensiveBids)),allRealOpponents:full.every(g=>g.opponents.length===5&&g.opponents.every(o=>!o.simulated)),perGame};
const file=resolve('../HERCULE-ONLINE-REVIEW.json');writeFileSync(file,JSON.stringify(result,null,2));
console.log(JSON.stringify({...result,perGame:undefined,file},null,2));
