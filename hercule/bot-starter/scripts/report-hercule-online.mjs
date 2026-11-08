import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,existsSync} from 'node:fs';
import {join,resolve} from 'node:path';
const json=f=>JSON.parse(readFileSync(f,'utf8').replace(/^\uFEFF/,''));
const activePath=resolve('runs/hercule-offense-online-active.json'),campaign=json(activePath),interval=campaign.reportIntervalGames??16;
const flag=process.argv.indexOf('--through'),through=flag>=0?Number(process.argv[flag+1]):(campaign.lastReportedFullGames??0)+interval;
assert.ok(Number.isInteger(through)&&through>=16&&through%interval===0);
const games=[],events=[];
for(const directory of campaign.runDirectories){
  const file=join(directory,'results.json');if(existsSync(file)){const r=json(file);assert.equal(r.policyVersion,'hercule-offense-v4');games.push(...r.results.map(g=>({...g,runDirectory:directory})));}
  const log=join(directory,'events.jsonl');if(existsSync(log))for(const line of readFileSync(log,'utf8').trim().split('\n')){try{events.push({...JSON.parse(line),runDirectory:directory});}catch{}}
}
const unique=[...new Map(games.map(g=>[g.matchId,g])).values()],full=unique.filter(g=>g.phase!=='partial');assert.ok(full.length>=through,'Checkpoint not reached');
const cumulative=full.slice(0,through),batch=cumulative.slice(-interval),ids=new Set(cumulative.map(g=>g.matchId));
for(const g of cumulative){assert.equal(g.words.reduce((s,w)=>s+w.reward,0),g.score,`Score/word mismatch ${g.matchId}`);assert.equal(g.rank,1+g.opponents.filter(o=>o.score>g.score).length);}
const mean=a=>a.length?a.reduce((s,n)=>s+n,0)/a.length:null;
function stats(rows){const words=rows.flatMap(g=>g.words),long=words.filter(w=>w.word.length>=7);return {games:rows.length,wins:rows.filter(g=>g.rank===1).length,winRate:rows.filter(g=>g.rank===1).length/rows.length,meanScore:mean(rows.map(g=>g.score)),meanRank:mean(rows.map(g=>g.rank)),meanWinnerScore:mean(rows.map(g=>Math.max(g.score,...g.opponents.map(o=>o.score)))),meanSpent:mean(rows.map(g=>g.totalSpend)),meanFinalCash:mean(rows.map(g=>100+g.score-g.totalSpend)),longWords:long.length,longWordsPerGame:long.length/rows.length,longWordMeanReward:mean(long.map(w=>w.reward)),longWordPoints:long.reduce((s,w)=>s+w.reward,0),longWordScoreShare:long.reduce((s,w)=>s+w.reward,0)/Math.max(1,rows.reduce((s,g)=>s+g.score,0)),realRivalCounts:[...new Set(rows.map(g=>g.opponents.filter(o=>!o.simulated).length))],simulatedRivalSeats:rows.reduce((s,g)=>s+g.opponents.filter(o=>o.simulated).length,0),defensiveBids:rows.reduce((s,g)=>s+g.defensiveBids,0),defensiveWins:rows.reduce((s,g)=>s+g.defensiveWins,0),defensiveSpend:rows.reduce((s,g)=>s+g.defensiveSpend,0)};}
const decisions=events.filter(e=>e.event==='decision'&&ids.has(String(e.data.matchId))),closed=events.filter(e=>e.event==='error'&&e.data.what==='bid'&&/Auction closed/i.test(e.data.message)),latencies=decisions.map(e=>e.data.elapsedMs).sort((a,b)=>a-b);
const connections=events.filter(e=>e.event==='connected'),stops=events.filter(e=>e.event==='stop');
const oldPath='runs/hercule-v1/2026-10-24T23-02-06-768Z/results.json',prior=existsSync(oldPath)?stats(json(oldPath).results.filter(g=>g.phase!=='partial')):null;
const result={reportedAt:new Date().toISOString(),through,blockStart:through-interval+1,policy:campaign.policy,configuration:campaign.configuration,batch:stats(batch),cumulative:stats(cumulative),priorEight:prior,excludedPartialGames:unique.filter(g=>g.phase==='partial').length,dataQuality:{fullGameDecisions:decisions.length,nonpositiveLocalTimeRemaining:decisions.filter(e=>e.data.observedRemainingMs<=0).length,policyLatencyP95Ms:latencies[Math.floor((latencies.length-1)*.95)],policyLatencyMaxMs:latencies.at(-1),auctionClosedErrorsCampaign:closed.length,connections:connections.length,stops:stops.map(e=>({time:e.time,...e.data})),note:'Closed-auction errors are campaign-wide, including partial games; error events do not contain match IDs. Local deadline estimates can be affected by clock differences.'},games:batch,cumulativeMatchIds:cumulative.map(g=>g.matchId),runDirectories:campaign.runDirectories};
const name=`ONLINE-CHECKPOINT-${String(through).padStart(4,'0')}`,directory=campaign.runDirectories[0],report=join(directory,`${name}.md`),artifact=join(directory,`${name}.json`);
writeFileSync(artifact,JSON.stringify(result,null,2));
const f=n=>n===null?'n/a':n.toFixed(2),a=result.batch,c=result.cumulative,p=result.priorEight;
const lines=[`# hercule-v1 online checkpoint: ${through} full games`,'',`Policy: ${campaign.policy}, ${campaign.configuration}, zero defense. Generated ${result.reportedAt}. The bot continues playing.`, '',
`This checkpoint covers games ${result.blockStart}–${through}. ${result.excludedPartialGames} recorded partial games are excluded. The earlier eight-game defensive-policy run is a separate descriptive comparison.`, '',
'| Metric | New block | Cumulative | Earlier eight |','|---|---:|---:|---:|',
`| Completed full games | ${a.games} | ${c.games} | ${p?.games??'n/a'} |`,
`| Shared-first wins | ${a.wins} (${f(a.winRate*100)}%) | ${c.wins} (${f(c.winRate*100)}%) | ${p?.wins??'n/a'} |`,
`| Mean own score | ${f(a.meanScore)} | ${f(c.meanScore)} | ${p?f(p.meanScore):'n/a'} |`,
`| Mean rank | ${f(a.meanRank)} | ${f(c.meanRank)} | ${p?f(p.meanRank):'n/a'} |`,
`| Mean winner score | ${f(a.meanWinnerScore)} | ${f(c.meanWinnerScore)} | ${p?f(p.meanWinnerScore):'n/a'} |`,
`| 7+ letter words per game | ${f(a.longWordsPerGame)} | ${f(c.longWordsPerGame)} | ${p?f(p.longWordsPerGame):'n/a'} |`,
`| Mean points per 7+ letter word | ${f(a.longWordMeanReward)} | ${f(c.longWordMeanReward)} | ${p?f(p.longWordMeanReward):'n/a'} |`,
`| Mean letter spend | ${f(a.meanSpent)} | ${f(c.meanSpent)} | ${p?f(p.meanSpent):'n/a'} |`,
`| Mean final cash | ${f(a.meanFinalCash)} | ${f(c.meanFinalCash)} | ${p?f(p.meanFinalCash):'n/a'} |`,'',
`Real rival counts per game in this block: ${a.realRivalCounts.join(', ')}. Simulated rival seats: ${a.simulatedRivalSeats}. Defensive bids/purchases/spend: ${a.defensiveBids}/${a.defensiveWins}/${a.defensiveSpend}.`, '',
'## Results by match','', '| Match | Score | Rank | Winner score | 7+ words | 7+ points |','|---|---:|---:|---:|---:|---:|',
...batch.map(g=>{const long=g.words.filter(w=>w.word.length>=7);return `| ${g.matchId} | ${g.score} | ${g.rank}/${g.opponents.length+1} | ${Math.max(g.score,...g.opponents.map(o=>o.score))} | ${long.length} | ${long.reduce((s,w)=>s+w.reward,0)} |`;}),'',
'## Data quality and interpretation','',
`${result.dataQuality.auctionClosedErrorsCampaign} bid requests were rejected as “Auction closed” across the campaign logs, including partial games. Of ${decisions.length} opening decisions in the checkpoint's full games, ${result.dataQuality.nonpositiveLocalTimeRemaining} had nonpositive local deadline estimates. These estimates alone cannot distinguish delivery delay from clock differences. Policy calculation time was ${f(result.dataQuality.policyLatencyP95Ms)} ms at the 95th percentile and ${f(result.dataQuality.policyLatencyMaxMs)} ms maximum.`, '',
`There have been ${connections.length} account connections. A connection drop was restarted with unchanged strategy and preserved logs. A zero-score full game, if present, remains included rather than being discarded. These delivery issues limit conclusions about strategy quality.`, '',
'This is a small observational online sample with different bags and opponents from the earlier run. A win does not establish improvement, and a lower score does not isolate the effect of removing defense or adding word preferences. The same policy remains running as authorized; no tuning or strategy change was performed.','',
`Structured snapshot: [${name}.json](${artifact.replaceAll('\\','/')}).`,''];
writeFileSync(report,lines.join('\n'));writeFileSync(resolve('../ONLINE-OFFENSE-RESULTS.md'),lines.join('\n'));
if(process.argv.includes('--mark-reported')){const latest=json(activePath);latest.lastReportedFullGames=Math.max(latest.lastReportedFullGames??0,through);latest.lastReportedAt=result.reportedAt;latest.lastReport=report;writeFileSync(activePath,JSON.stringify(latest,null,2));}
console.log(JSON.stringify({report,artifact,through,batch:result.batch,cumulative:result.cumulative,prior:result.priorEight,dataQuality:result.dataQuality,markedReported:process.argv.includes('--mark-reported')},null,2));
