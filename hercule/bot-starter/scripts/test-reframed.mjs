// Offline paired policy comparison. Imports no connection or credential code.
import {readFileSync,writeFileSync,mkdirSync} from "node:fs";
import {resolve,join} from "node:path";
import {createHash} from "node:crypto";
import {loadDictionary} from "../src/dictionary.ts";
import {chooseWord,prepareDictionary} from "../src/strategy.ts";
import {makeWagsAgent,makeHerculeAgent,makeModelAgent,simulateMatch} from "../src/local-arena.ts";
import {decideWagsBid,chooseWagsWord} from "./fixtures/wags-v1.ts";
import {defensiveBid} from "./fixtures/defense-v1.ts";
import {DEFAULT_BAG} from "../../spacetimedb/src/letters.ts";
import {SCORING_POLICY_VERSION} from "../src/scoring-v3.ts";

const args=process.argv.slice(2);
const option=(name,fallback)=>{const i=args.indexOf(name);return i<0?fallback:args[i+1];};
const seeds=Number(option("--seeds",100)),seedOffset=Number(option("--seed-offset",3000000));
if(!Number.isInteger(seeds)||seeds<1||seeds>1000||!Number.isSafeInteger(seedOffset)||seedOffset<0)throw new Error("Invalid seed options.");
const label=option("--label","comparison");
if(!/^[a-z0-9-]+$/i.test(label))throw new Error("Invalid label.");
const startedAt=new Date().toISOString(),output=resolve("local-tests",`reframed-${label}-${startedAt.replace(/[:.]/g,"-")}`);
mkdirSync(output,{recursive:true});
const dictionary=loadDictionary();prepareDictionary(dictionary);
const bag=DEFAULT_BAG.flatMap(([letter,n])=>Array(n).fill(letter));
const summary=JSON.parse(readFileSync("opponents/summary.json","utf8"));
const models=summary.selected.filter(model=>model.tier==="priority").slice(0,5).map(model=>JSON.parse(readFileSync(`opponents/models/bot-${model.botId}.json`,"utf8")));
if(models.length!==5)throw new Error("Five cached opponent models required.");
const rivals=models.map(makeModelAgent);
const legacy={
  wags:{name:"Wags v1",kind:"wags",bid:decideWagsBid,word:chooseWagsWord},
  hercule:{name:"hercule-v1",kind:"hercule",bid:ctx=>defensiveBid(ctx,ctx.opponents,1.6).amount,word:chooseWord},
};
const revised={wags:makeWagsAgent(),hercule:makeHerculeAgent(1.6)};
const records=[],paired={wags:[],hercule:[]};
const started=performance.now();
console.log(`Offline comparison: ${4*seeds} six-player games; ${seeds} matched bags per bot; no network.`);
for(let i=0;i<seeds;i++)for(const kind of ["wags","hercule"]){
  const pair={};
  for(const version of ["before","after"]){
    const target=(version==="before"?legacy:revised)[kind],agents=[...rivals];agents.splice(i%6,0,target);
    const game=simulateMatch({seed:seedOffset+i,agents,bag,dictionary});
    const own=game.players[i%6],top=Math.max(...game.players.filter((_,seat)=>seat!==i%6).map(p=>p.score));
    pair[version]={...own,topRival:top};records.push({kind,version,...game});
  }
  paired[kind].push({seed:seedOffset+i,...pair});
  if(records.length%20===0){
    console.log(`${records.length}/${4*seeds} (${((performance.now()-started)/1000).toFixed(1)}s)`);
    writeFileSync(join(output,"progress.json"),JSON.stringify({games:records.length,total:4*seeds}));
  }
}
const mean=a=>a.reduce((s,n)=>s+n,0)/Math.max(1,a.length);
function interval(a){const average=mean(a),variance=a.reduce((s,n)=>s+(n-average)**2,0)/Math.max(1,a.length-1),margin=a.length>1?1.96*Math.sqrt(variance/a.length):null;return {mean:average,interval95:margin===null?null:[average-margin,average+margin],independentBags:a.length};}
function aggregate(rows){return {games:rows.length,score:mean(rows.map(p=>p.score)),topRival:mean(rows.map(p=>p.topRival)),winRate:mean(rows.map(p=>Number(p.sharedWin))),rank:mean(rows.map(p=>p.rank)),spent:mean(rows.map(p=>p.spent)),purchases:mean(rows.map(p=>p.purchases)),endingCash:mean(rows.map(p=>p.balance)),wordLength:rows.reduce((s,p)=>s+p.wordLetters,0)/Math.max(1,rows.reduce((s,p)=>s+p.words,0)),longScoreFraction:rows.reduce((s,p)=>s+p.longScore,0)/Math.max(1,rows.reduce((s,p)=>s+p.score,0))};}
const policies=Object.fromEntries(Object.entries(paired).map(([kind,pairs])=>[kind,{
  before:aggregate(pairs.map(p=>p.before)),after:aggregate(pairs.map(p=>p.after)),
  scoreChange:interval(pairs.map(p=>p.after.score-p.before.score)),
  topRivalChange:interval(pairs.map(p=>p.after.topRival-p.before.topRival)),
}]));
const sources=["src/strategy.ts","src/planning.ts","src/competition.ts","src/scoring-v3.ts","src/leader-priors-v3.ts","src/intel.ts","src/defense.ts","src/wags.ts","src/local-arena.ts","src/public-opponents.ts","src/online.ts","src/study.ts","scripts/test-reframed.mjs","scripts/fixtures/wags-v1.ts","scripts/fixtures/defense-v1.ts","../spacetimedb/wordlist.txt","../spacetimedb/src/letters.ts",...models.map(model=>`opponents/models/bot-${model.botId}.json`)];
const result={policyVersion:SCORING_POLICY_VERSION,startedAt,completedAt:new Date().toISOString(),elapsedSeconds:(performance.now()-started)/1000,seeds,seedOffset,games:records.length,models:models.map(m=>({name:m.name,collectedAt:m.collectedAt,fidelity:m.evidence.fidelity})),policies,sources:sources.map(file=>({file,sha256:createHash("sha256").update(readFileSync(file)).digest("hex")}))};
writeFileSync(join(output,"results.json"),JSON.stringify(result,null,2));
writeFileSync(join(output,"matches.json"),JSON.stringify(records));
const fmt=n=>n.toFixed(1);
const report=["# Reframed policy comparison","",`${result.games} offline six-player matches, ${seeds} matched independent bags per bot. Target seats rotate; each before/after pair has identical opponents, tile order, target seat, and tie randomness. hercule-v1 uses weight 1.6 in both versions. No live matches, credentials, or network access.`,"",
"| Bot | Policy | Average score | Average strongest rival | Win rate | Purchases | Average word length | Points from 7+ letters |","|---|---|---:|---:|---:|---:|---:|---:|",
...Object.entries(policies).flatMap(([kind,p])=>["before","after"].map(version=>{const s=p[version];return `| ${kind} | ${version} | ${fmt(s.score)} | ${fmt(s.topRival)} | ${fmt(100*s.winRate)}% | ${fmt(s.purchases)} | ${fmt(s.wordLength)} | ${fmt(100*s.longScoreFraction)}% |`;})),"",
...Object.entries(policies).map(([kind,p])=>`${kind}: paired score change ${fmt(p.scoreChange.mean)}, approximate 95% interval ${JSON.stringify(p.scoreChange.interval95)}. Strongest-rival change ${fmt(p.topRivalChange.mean)}, interval ${JSON.stringify(p.topRivalChange.interval95)}; negative means greater suppression.`),"",
"These are comparisons against five cached approximate public-data models, not the real bots. They omit newer contenders, have documented prediction errors, and idealize network timing equally. Intervals describe bag variation, not opponent-model error. Shared first places count as wins. Both old and new policies see only their own dealt letters, public rack sizes, retained public purchases, balances, and scores. All matches check cash/tile conservation and forbid post-end cashing of the final awarded tile.","",
`Seed offset ${seedOffset}. Runtime ${fmt(result.elapsedSeconds)} seconds. Source and model hashes: results.json. Complete games: matches.json.`,""].join("\n");
writeFileSync(join(output,"REPORT.md"),report);
console.log(JSON.stringify({directory:output,...result},null,2));
