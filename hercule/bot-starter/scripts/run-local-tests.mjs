import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { DEFAULT_BAG,letterValue,lengthMultiplier } from "../../spacetimedb/src/letters.ts";
import { loadDictionary } from "../src/dictionary.ts";
import { prepareDictionary,wordReward } from "../src/strategy.ts";
import { prepareWagsDictionary } from "../src/wags.ts";
import {SCORING_POLICY_VERSION} from "../src/scoring-v3.ts";
import {HERCULE_LIVE_VERSION as HERCULE_POLICY_VERSION,HERCULE_LIVE_CONFIG as HERCULE_ACTIVE_CONFIG,makeLiveHerculeAgent} from '../src/hercule-live.ts';
import { makeWagsAgent,makeHerculeAgent,makeModelAgent,simulateMatch } from "../src/local-arena.ts";

const args = process.argv.slice(2);
const option = (flag,fallback) => { const i=args.indexOf(flag);return i>=0?args[i+1]:fallback; };
const seeds = Number(option("--seeds",200));
const seedOffset = Number(option("--seed-offset",0));
if (!Number.isInteger(seeds) || seeds<1 || seeds>5000) throw new Error("--seeds must be 1–5000.");
if (!Number.isSafeInteger(seedOffset) || seedOffset<0 || seedOffset>2**31) throw new Error("Invalid seed offset.");
const startedAt = new Date().toISOString();
const output = resolve("local-tests",startedAt.replace(/[:.]/g,"-"));
mkdirSync(output,{recursive:true});
const dictionary = loadDictionary();
prepareDictionary(dictionary); prepareWagsDictionary(dictionary);
// Check the scoring implementation against the game's pure rule functions,
// including odd half-point bonuses, before any tournament is simulated.
for (const word of ["AX","JAZZ","QUIZ","RATES","RETAIN","RETAINS"]) {
  const [num,denom] = lengthMultiplier(word.length);
  const base = [...word].reduce((sum,letter)=>sum+BigInt(letterValue(letter)),0n);
  assert.equal(wordReward(word),Number(base*num/denom));
}
const bag = DEFAULT_BAG.flatMap(([letter,count])=>Array(count).fill(letter));
assert.equal(bag.length,98);
const summary = JSON.parse(readFileSync("opponents/summary.json","utf8"));
const models = summary.selected.filter((model)=>model.tier==="priority").slice(0,5)
  .map((model)=>JSON.parse(readFileSync(`opponents/models/bot-${model.botId}.json`,"utf8")));
if (models.length!==5) throw new Error("Five priority public-data models are required; build them first.");
const defenseWeight=0;
const wags = makeWagsAgent(), hercule = makeLiveHerculeAgent();
const opponents = models.map(makeModelAgent);
const records = [];
const expectedGames = 4*seeds+2*Math.ceil(seeds/2);
const start = performance.now();

function run(scenario,seed,placement,agents) {
  const result = simulateMatch({seed,agents,dictionary,bag});
  records.push({scenario,placement,...result});
  if (records.length%25===0) console.log(`${records.length}/${expectedGames} local matches (${((performance.now()-start)/1000).toFixed(1)}s)`);
  if (records.length%50===0) writeFileSync(join(output,"progress.json"),JSON.stringify({completed:records.length,total:expectedGames,elapsedSeconds:(performance.now()-start)/1000}));
}

console.log(`Offline evaluation: ${expectedGames} games, five private starting tiles, Vickrey auctions, no network.`);
console.log(`hercule-v1 ${HERCULE_POLICY_VERSION}, defense disabled; models: ${models.map((model)=>model.name).join(", ")}.`);

for (let i=0;i<seeds;i++) {
  const seed = seedOffset+10000+i;
  const first = i%6, second = (first+1+i%5)%6;
  const roster = opponents.filter((_opponent,index)=>index!==i%5);
  for (let swap=0;swap<2;swap++) {
    const agents = Array(6);
    agents[first] = swap?hercule:wags;
    agents[second] = swap?wags:hercule;
    let j=0; for (let seat=0;seat<6;seat++) if (!agents[seat]) agents[seat]=roster[j++];
    run("mixed",seed,swap,agents);
  }
}
for (let i=0;i<seeds;i++) {
  const seed = seedOffset+30000+i, seat = i%6;
  for (const target of [wags,hercule]) {
    const agents = [...opponents]; agents.splice(seat,0,target);
    run("replacement",seed,target.kind,agents);
  }
}
for (let i=0;i<Math.ceil(seeds/2);i++) {
  const seed = seedOffset+50000+i;
  run("duel",seed,0,[wags,hercule]);
  run("duel",seed,1,[hercule,wags]);
}

const mean = (values)=>values.reduce((a,b)=>a+b,0)/Math.max(1,values.length);
function interval(values) {
  const average = mean(values);
  const variance = values.reduce((sum,value)=>sum+(value-average)**2,0)/Math.max(1,values.length-1);
  const margin = values.length>1?1.96*Math.sqrt(variance/values.length):null;
  return {mean:average,interval95:margin===null?null:[average-margin,average+margin],independentSeeds:values.length};
}
function aggregate(scenario,kind) {
  const samples = records.filter((game)=>game.scenario===scenario).flatMap((game)=>game.players.filter((player)=>player.kind===kind));
  return {games:samples.length,score:mean(samples.map((p)=>p.score)),winRate:mean(samples.map((p)=>Number(p.sharedWin))),
    rank:mean(samples.map((p)=>p.rank)),spent:mean(samples.map((p)=>p.spent)),purchases:mean(samples.map((p)=>p.purchases)),
    words:mean(samples.map((p)=>p.words)),averageWordLength:samples.reduce((sum,p)=>sum+p.wordLetters,0)/Math.max(1,samples.reduce((sum,p)=>sum+p.words,0)),
    shortWordRate:samples.reduce((sum,p)=>sum+p.shortWords,0)/Math.max(1,samples.reduce((sum,p)=>sum+p.words,0)),balance:mean(samples.map((p)=>p.balance))};
}
const mixedBySeed = new Map();
for (const game of records.filter((game)=>game.scenario==="mixed")) {
  const delta = game.players.find((p)=>p.kind==="wags").score-game.players.find((p)=>p.kind==="hercule").score;
  if (!mixedBySeed.has(game.seed)) mixedBySeed.set(game.seed,[]); mixedBySeed.get(game.seed).push(delta);
}
const replacementBySeed = new Map();
for (const game of records.filter((game)=>game.scenario==="replacement")) {
  if (!replacementBySeed.has(game.seed)) replacementBySeed.set(game.seed,{});
  replacementBySeed.get(game.seed)[game.placement]=game;
}
const replacementScore = [],replacementRival = [];
for (const pair of replacementBySeed.values()) {
  replacementScore.push(pair.wags.players.find((p)=>p.kind==="wags").score-pair.hercule.players.find((p)=>p.kind==="hercule").score);
  replacementRival.push(Math.max(...pair.wags.players.filter((p)=>p.kind==="model").map((p)=>p.score))-
    Math.max(...pair.hercule.players.filter((p)=>p.kind==="model").map((p)=>p.score)));
}
const duels = records.filter((game)=>game.scenario==="duel");
const duelCounts = {wags:0,hercule:0,ties:0};
for (const game of duels) {
  const w = game.players.find((p)=>p.kind==="wags"),t = game.players.find((p)=>p.kind==="hercule");
  if (w.score===t.score) duelCounts.ties++; else if(w.score>t.score) duelCounts.wags++; else duelCounts.hercule++;
}
const sourceFiles = ["scripts/run-local-tests.mjs","src/hercule-live.ts","src/hercule-median-settings.ts","src/median-policy.ts","src/value-planning.ts","src/hercule.ts","src/hercule-policy.ts","src/hercule-settings.ts","src/hercule-priors.ts","src/strategy.ts","src/planning.ts","src/competition.ts","src/scoring-v3.ts","src/leader-priors-v3.ts","src/intel.ts","src/wags.ts","src/defense.ts","src/local-arena.ts",...models.map((m)=>`opponents/models/bot-${m.botId}.json`),"../spacetimedb/wordlist.txt","../spacetimedb/src/letters.ts"];
const result = {startedAt,completedAt:new Date().toISOString(),elapsedSeconds:(performance.now()-start)/1000,gameCount:records.length,
  randomGenerator:"mulberry32",seedOffset,independentSeedCounts:{mixed:seeds,replacement:seeds,duel:Math.ceil(seeds/2)},defenseWeight,policyVersions:{wags:SCORING_POLICY_VERSION,hercule:HERCULE_POLICY_VERSION},herculeConfiguration:HERCULE_ACTIVE_CONFIG,auctionType:"Vickrey",startingBalance:100,startingRackSize:5,
  models:models.map((m)=>({name:m.name,botId:m.botId,snapshot:m.collectedAt,fidelity:m.evidence.fidelity})),
  mixed:{wags:aggregate("mixed","wags"),hercule:aggregate("mixed","hercule"),pairedScoreDifference:interval([...mixedBySeed.values()].map(mean))},
  replacement:{wags:aggregate("replacement","wags"),hercule:aggregate("replacement","hercule"),pairedScoreDifference:interval(replacementScore),
    strongestRivalDifference:interval(replacementRival)},
  duel:{wags:aggregate("duel","wags"),hercule:aggregate("duel","hercule"),outcomes:duelCounts},
  sources:sourceFiles.map((file)=>({file,sha256:createHash("sha256").update(readFileSync(file)).digest("hex")}))};
writeFileSync(join(output,"results.json"),JSON.stringify(result,null,2));
writeFileSync(join(output,"matches.json"),JSON.stringify(records));
const fmt = (value)=>value.toFixed(1);
const ci = (value)=>value.interval95?`${fmt(value.mean)} [${fmt(value.interval95[0])}, ${fmt(value.interval95[1])}]`:`${fmt(value.mean)} (insufficient independent seeds for interval)`;
const metricRow = (name,stats)=>`| ${name} | ${stats.games} | ${fmt(stats.score)} | ${(100*stats.winRate).toFixed(1)}% | ${fmt(stats.rank)} | ${fmt(stats.spent)} | ${fmt(stats.words)} | ${fmt(stats.averageWordLength)} |`;
const report = ["# Wags and hercule-v1: local match comparison","",
  `${records.length} deterministic local games completed across three scenarios. No network, account tokens, or public match entry was used. Neither live strategy nor its saved parameters was tuned by this evaluation.`,"",
  `hercule-v1 uses ${HERCULE_POLICY_VERSION} with defense disabled. Wags retains ${SCORING_POLICY_VERSION}, aggressive acquisition, planned payouts, and contender pressure. Model opponents: ${models.map((m)=>m.name).join(", ")}.`,"",
  "## Six-player games with both bots","",
  `${2*seeds} games on ${seeds} independent tile bags, with Wags and hercule-v1 swapping starting positions for each bag. Four of the five priority models fill the remaining seats, with the omitted model rotated.`,"",
  "| Bot | Games | Avg score | Shared-first win rate | Avg rank | Avg spend | Words/game | Avg word length |",
  "|---|---:|---:|---:|---:|---:|---:|---:|",metricRow("Wags",result.mixed.wags),metricRow("hercule-v1",result.mixed.hercule),"",
  `Paired Wags − hercule-v1 score difference, averaged within each seed before calculating the interval: **${ci(result.mixed.pairedScoreDifference)}**.`,"",
  "## Replacement comparison against five models","",
  `${2*seeds} games: ${seeds} matched pairs, replacing only the target bot in the same starting seat and keeping the letter bag and opponent roster fixed.`,"",
  "| Bot | Games | Avg score | Shared-first win rate | Avg rank | Avg spend | Words/game | Avg word length |",
  "|---|---:|---:|---:|---:|---:|---:|---:|",metricRow("Wags",result.replacement.wags),metricRow("hercule-v1",result.replacement.hercule),"",
  `Paired target-score difference Wags − hercule-v1: **${ci(result.replacement.pairedScoreDifference)}**.`,
  `Highest modeled rival score with Wags − highest modeled rival score with hercule-v1: **${ci(result.replacement.strongestRivalDifference)}**. Negative values mean the modeled field scored less with Wags; positive values favor hercule-v1's suppression.`,"",
  "## Two-player duels","",
  `${duels.length} games, ${Math.ceil(seeds/2)} independent tile bags with seats swapped. Strict wins: Wags ${duelCounts.wags}, hercule-v1 ${duelCounts.hercule}; ties ${duelCounts.ties}. Two-player economics differ from the six-player competition format.`,"",
  "## Rules, checks, and limitations","",
  "Uses the source game's standard 98-tile distribution, five private starting tiles per player, 100 starting coins, Vickrey payment with reserve 1, randomized tie ordering, dictionary validation, and integer-rounded word rewards. Wins include shared first-place finishes. Every match checks cash and tile conservation. Only a player's own rack and public score/balance/retained purchases are supplied to its strategy; initial opponent letters and sealed bids are hidden.","",
  "Opening bids are submitted before words are cashed during the auction. Existing words can be played repeatedly within the auction window; the final awarded tile cannot be cashed after match end. RPC latency and client timing differences are idealized equally for all policies. No timer or lobby delay is simulated.","",
  "The public-data opponents are approximate behavior models. Several have poor losing-bid fidelity, and their own multiplayer strength has not been validated. These results compare policies inside this simulator, not their actual online win rates. The approximate 95% normal intervals describe variation across independent local seeds and do not cover model mismatch or network timing uncertainty.","",
  `Local duration: ${fmt(result.elapsedSeconds)} seconds. Complete per-game data: ${join(output,"matches.json")}. Aggregate data and source hashes: ${join(output,"results.json")}.`,"",
  `Reproduce this batch from bot-starter: \`npm.cmd run test:local -- --seeds ${seeds} --seed-offset ${seedOffset}\`. It never connects online. The shuffle uses a mixed seeded generator so consecutive seed inputs do not constrain the first shuffle draw to a narrow range.`,"",
].join("\n");
writeFileSync(join(output,"REPORT.md"),report);
writeFileSync(resolve("../LOCAL-TEST-RESULTS.md"),report);
console.log(JSON.stringify({report:resolve("../LOCAL-TEST-RESULTS.md"),...result},null,2));
