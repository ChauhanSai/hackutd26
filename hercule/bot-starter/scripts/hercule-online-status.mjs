import {readFileSync,existsSync,writeFileSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
const activePath=resolve('runs/hercule-offense-online-active.json');
const json=file=>JSON.parse(readFileSync(file,'utf8').replace(/^\uFEFF/,''));
if(!existsSync(activePath))throw new Error('No continuous hercule-v1 campaign recorded');
const launch=json(activePath),games=[],events=[];
for(const directory of launch.runDirectories??[]){
  const result=join(directory,'results.json');if(existsSync(result)){const r=json(result);if(r.policyVersion!=='hercule-offense-v4')throw new Error('Unexpected policy version');games.push(...r.results.map(g=>({...g,runDirectory:directory})));}
  const file=join(directory,'events.jsonl');if(existsSync(file))for(const line of readFileSync(file,'utf8').trim().split('\n')){try{events.push(JSON.parse(line));}catch{}}
}
const unique=[...new Map(games.map(g=>[g.matchId,g])).values()],full=unique.filter(g=>g.phase!=='partial'),partial=unique.filter(g=>g.phase==='partial');
let alive=false,processAccessRestricted=false;try{process.kill(launch.pid,0);alive=true;}catch(error){if(error.code==='EPERM'){alive=true;processAccessRestricted=true;}}
const last=events.at(-1),connected=events.filter(e=>e.event==='connected').at(-1),stop=events.filter(e=>e.event==='stop').at(-1),errors=events.filter(e=>e.event==='error');
const changedSources=(launch.sourceHashes??[]).filter(s=>createHash('sha256').update(readFileSync(s.file)).digest('hex')!==s.sha256).map(s=>s.file);
const first=full.slice(0,launch.firstReportGames),recent=full.slice(-16),mean=a=>a.length?a.reduce((s,n)=>s+n,0)/a.length:null;
function summarize(rows){return {games:rows.length,wins:rows.filter(g=>g.rank===1).length,meanScore:mean(rows.map(g=>g.score)),meanRank:mean(rows.map(g=>g.rank)),meanWinnerScore:mean(rows.map(g=>Math.max(g.score,...g.opponents.map(o=>o.score)))),realRivalCounts:[...new Set(rows.map(g=>g.opponents.filter(o=>!o.simulated).length))],defensiveBids:rows.reduce((s,g)=>s+g.defensiveBids,0),defensiveWins:rows.reduce((s,g)=>s+g.defensiveWins,0),matches:rows.map(g=>({matchId:g.matchId,score:g.score,rank:g.rank,opponents:g.opponents.length,realRivals:g.opponents.filter(o=>!o.simulated).length}))};}
const status={checkedAt:new Date().toISOString(),pid:launch.pid,alive,processAccessRestricted,connected:!!connected,connectedAt:connected?.time,fullGames:full.length,partialGames:partial.length,firstReportGames:launch.firstReportGames,reportDue:full.length>=launch.firstReportGames&&full.length-(launch.lastReportedFullGames??0)>=(launch.lastReportedFullGames?launch.reportIntervalGames:launch.firstReportGames),lastReportedFullGames:launch.lastReportedFullGames??0,lastEvent:last?.time,lastEventType:last?.event,stop:stop?.data,errors:errors.length,lastError:errors.at(-1),changedSources,firstBatch:summarize(first),recentBatch:summarize(recent),cumulative:summarize(full),runDirectory:launch.runDirectory};
if(process.argv.includes('--save'))writeFileSync(resolve('runs/hercule-offense-online-status.json'),JSON.stringify(status,null,2));
if(process.argv.includes('--compact')){for(const key of ['firstBatch','recentBatch','cumulative'])delete status[key];}
console.log(JSON.stringify(status,null,2));
