import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { chronologicalSplit, fitBid, evaluateBid, quantile, wilsonInterval } from "../src/public-opponents.ts";

const { directory } = JSON.parse(readFileSync("opponents/latest-snapshot.json", "utf8"));
const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
function table(name) {
  const payload = JSON.parse(readFileSync(join(directory, `${name}.json`), "utf8"))[0];
  const columns = payload.schema.elements.map((field) => field.name.some);
  return payload.rows.map((values) => Object.fromEntries(columns.map((column, i) => [column, values[i]])));
}
const group = (rows, key) => {
  const result = new Map();
  for (const row of rows) { if (!result.has(row[key])) result.set(row[key], []); result.get(row[key]).push(row); }
  return result;
};
const stamp = (value) => Number(value[0]);
const matches = new Map(table("match_state").map((match) => [match.id, match]));
const auctions = group(table("auction_result"), "match_id");
const words = group(table("word_play"), "match_id");
const participants = group(table("match_participant"), "match_id");
const topBids = new Map(table("auction_top_bid_archive").map((row) => [row.auction_id, row.top_bid]));
const selectedIds = new Map(manifest.selected.map((bot) => [bot.botId, new Set(bot.matchIds)]));
const observations = new Map(manifest.selected.map((bot) => [bot.botId, []]));
const wordObservations = new Map(manifest.selected.map((bot) => [bot.botId, []]));
const audits = [];
let skippedLateBalance = 0;

for (const matchId of manifest.selectedMatchIds) {
  const match = matches.get(matchId);
  const roster = participants.get(matchId) ?? [];
  const rounds = [...(auctions.get(matchId) ?? [])].sort((a,b) => stamp(a.closed_at)-stamp(b.closed_at) || a.auction_id-b.auction_id);
  const plays = [...(words.get(matchId) ?? [])].sort((a,b) => stamp(a.played_at)-stamp(b.played_at) || a.id-b.id);
  const targets = manifest.selected.filter((bot) => selectedIds.get(bot.botId).has(matchId));
  const ledgers = new Map(roster.map((row) => [row.bot_id, { balance: 100, score: 0, size: match.starting_rack_size, known: new Map(), invalid: false }]));
  const samples = new Map(targets.map((bot) => [bot.botId, []]));
  const wordSamples = new Map(targets.map((bot) => [bot.botId, []]));
  let nextWord = 0;
  const applyWords = (time, inclusive) => {
    while (nextWord < plays.length && (stamp(plays[nextWord].played_at) < time || inclusive && stamp(plays[nextWord].played_at) === time)) {
      const play = plays[nextWord++];
      const state = ledgers.get(play.bot_id);
      if (!state) continue;
      if (wordSamples.has(play.bot_id)) wordSamples.get(play.bot_id).push({ matchId, word: play.word, length: play.word.length, balanceBefore: state.balance, reward: play.total_reward });
      state.balance += play.total_reward;
      state.score += play.total_reward;
      state.size -= play.word.length;
      if (state.size < 0) state.invalid = true;
      for (const letter of play.word) state.known.set(letter, Math.max(0, (state.known.get(letter) ?? 0) - 1));
    }
  };
  let openTime = match.started_at[0] === 0 ? Number(match.started_at[1][0]) : stamp(match.created_at);
  for (let round = 0; round < rounds.length; round++) {
    const auction = rounds[round];
    applyWords(openTime, true);
    const winningId = auction.winner_bot_id[0] === 0 ? auction.winner_bot_id[1] : undefined;
    const exactTop = topBids.get(auction.auction_id) ?? (match.auction_type[0] === 0 ? auction.paid : undefined);
    for (const bot of targets) {
      const state = ledgers.get(bot.botId);
      if (!state || state.balance <= 0 || state.invalid) continue;
      const won = winningId === bot.botId;
      const kind = won ? exactTop !== undefined ? "exact" : "lower" : "upper";
      const amount = won ? exactTop ?? auction.paid : winningId === undefined ? 0 : exactTop;
      if (amount === undefined || kind === "upper" && amount >= state.balance) continue;
      // Some clients revise bids after cashing a word during the auction.
      // Such an exact bid cannot be explained from the opening balance alone.
      if (kind !== "upper" && amount > state.balance) { skippedLateBalance++; continue; }
      samples.get(bot.botId).push({ matchId, letter: auction.letter, balance: state.balance, score: state.score,
        rackSize: state.size, knownCopies: state.known.get(auction.letter) ?? 0,
        progress: round / Math.max(1, rounds.length - 1), leaderScore: Math.max(...[...ledgers.values()].map((s) => s.score)),
        firstPrice: match.auction_type[0] === 0, kind, amount });
    }
    const closeTime = stamp(auction.closed_at);
    applyWords(closeTime, false);
    const winner = ledgers.get(winningId);
    if (winner) {
      winner.balance -= auction.paid;
      winner.size++;
      winner.known.set(auction.letter, (winner.known.get(auction.letter) ?? 0) + 1);
      if (winner.balance < 0) winner.invalid = true;
    }
    applyWords(closeTime, true);
    openTime = closeTime;
  }
  applyWords(Infinity, true);
  const errors = roster.filter((row) => {
    const state = ledgers.get(row.bot_id);
    return state.invalid || state.balance !== row.balance || state.score !== row.score;
  }).map((row) => row.bot_id);
  const valid = !errors.length && rounds.length > 0;
  audits.push({ matchId, valid, rounds: rounds.length, declaredStartingRackSize: match.starting_rack_size, errorBots: errors });
  if (!valid) continue;
  for (const bot of targets) {
    observations.get(bot.botId).push(...samples.get(bot.botId));
    wordObservations.get(bot.botId).push(...wordSamples.get(bot.botId));
  }
}

const models = [];
mkdirSync("opponents/models", { recursive: true });
for (const bot of manifest.selected) {
  const samples = observations.get(bot.botId);
  const validMatchIds = bot.matchIds.filter((id) => audits.find((audit) => audit.matchId === id)?.valid);
  if (validMatchIds.length < 5) { console.log(`${bot.name}: insufficient audited games (${validMatchIds.length})`); continue; }
  const split = chronologicalSplit(validMatchIds);
  const trainIds = new Set(split.train), testIds = new Set(split.test);
  const train = samples.filter((sample) => trainIds.has(sample.matchId));
  const test = samples.filter((sample) => testIds.has(sample.matchId));
  const inner = chronologicalSplit(split.train);
  const innerTrain = new Set(inner.train), calibration = new Set(inner.test);
  const fitting = train.filter((sample) => innerTrain.has(sample.matchId));
  const tuning = train.filter((sample) => calibration.has(sample.matchId));
  const families = ["censored-linear", "letter-median"].map((family) => ({ family,
    loss: evaluateBid(fitBid(fitting, family), tuning).selectionLoss }));
  families.sort((a,b) => a.loss-b.loss);
  const bid = fitBid(train, families[0].family);
  const heldOut = evaluateBid(bid, test);
  const baseline = evaluateBid(fitBid(train, "letter-median"), test);
  const trainWords = wordObservations.get(bot.botId).filter((word) => trainIds.has(word.matchId));
  const testWords = wordObservations.get(bot.botId).filter((word) => testIds.has(word.matchId));
  const histogram = {};
  for (const word of trainWords) histogram[word.length] = (histogram[word.length] ?? 0) + 1;
  const modalLength = Number(Object.entries(histogram).sort((a,b) => b[1]-a[1] || Number(b[0])-Number(a[0]))[0]?.[0] ?? 2);
  const model = { schemaVersion: 1, name: bot.name, botId: bot.botId, version: bot.version, collectedAt: manifest.collectedAt, tier: bot.tier,
    leaderboard: { matches: bot.matches, wins: bot.wins, winRate: bot.winRate, interval95: wilsonInterval(bot.wins,bot.matches) }, bid,
    word: { minimumLength: Math.max(2,Math.min(10,modalLength)), observedModalLength: modalLength,
      policyBasis: "Aim for the observed modal word length; the target threshold and shorter-word fallback behavior are approximations.",
      trainingLengthHistogram: histogram, cashOutBalance: 20, flushRemaining: 3, rackLimit: 14, fallbackIsHeuristic: true },
    evidence: { sourceEndpoint: manifest.endpoint, snapshot: directory, trainMatchIds: split.train, testMatchIds: split.test,
      familySelection: { calibrationMatchIds: inner.test, scores: families },
      trainingSamples: train.length, trainingExactWinningBids: train.filter((sample) => sample.kind === "exact").length,
      trainingWordPlays: trainWords.length, heldOutWordPlays: testWords.length,
      heldOutMedianWordLength: quantile(testWords.map((word) => word.length),0.5), heldOut, letterMedianBaseline: baseline,
      fidelity: (heldOut.upperViolationRate ?? 1) > 0.25 ? "limited: frequent losing-bound conflicts" : "partial: hidden-rack and losing-bid uncertainty remains",
      limits: ["Only winning bids are observed exactly; losses supply upper bounds.",
        "Initial letters and unsubmitted bids are hidden. Opening-state features can miss late bid revisions.",
        "Word-length tendency is fitted; cash-out, endgame flushing, rack limit, and best-reward word selection are heuristic approximations.",
        "Prediction checks on historical games do not establish reproduced win rate or multiplayer strength."] },
  };
  writeFileSync(`opponents/models/bot-${bot.botId}.json`,JSON.stringify(model,null,2));
  models.push(model);
  console.log(`${bot.name}: ${split.train.length} train / ${split.test.length} newer holdout matches, ${bid.family}, bid MAE ${heldOut.winningBidMAE?.toFixed(2)}, upper-bound violations ${((heldOut.upperViolationRate??0)*100).toFixed(1)}%`);
}
const hashes = manifest.queries.map((query) => ({ ...query, sha256: createHash("sha256").update(readFileSync(join(directory,query.filename))).digest("hex") }));
const summary = { collectedAt: manifest.collectedAt, snapshot: directory, selected: manifest.selected, audit: { matches: audits.length,
  valid: audits.filter((audit) => audit.valid).length, invalid: audits.filter((audit) => !audit.valid), skippedLateBalance },
  models: models.map((model) => ({ botId:model.botId,name:model.name,tier:model.tier,leaderboard:model.leaderboard,
    minimumWordLength:model.word.minimumLength,family:model.bid.family,evidence:model.evidence })), sources:hashes };
writeFileSync("opponents/summary.json",JSON.stringify(summary,null,2));
const esc = (text) => String(text).replaceAll("|","\\|").replace(/[\r\n]/g," ");
const percent = (value) => `${(100*value).toFixed(1)}%`;
const reportRows = models.map((model) => `| ${esc(model.name)} | ${model.tier} | ${model.leaderboard.wins}/${model.leaderboard.matches} | ${percent(model.leaderboard.winRate)} | ${model.evidence.trainMatchIds.length}/${model.evidence.testMatchIds.length} | ${model.word.minimumLength} | ${model.evidence.heldOut.winningBidMAE?.toFixed(2)??"n/a"} | ${percent(model.evidence.heldOut.upperViolationRate??0)} |`);
const report = ["# Local opponents modeled from public server history", "",
  `Public snapshot collected ${new Date(manifest.collectedAt).toLocaleString("en-US",{timeZone:"America/Los_Angeles",timeZoneName:"short"})}. No bot token was used, no accounts or bot registrations were created, and no matches were entered or simulated.`, "",
  "## Selection and fitted models", "", manifest.selection,
  "", "Win percentages are for each bot's current leaderboard version and include shared first-place finishes. They are not adjusted for opponent strength or simulated filler seats; historical win rates do not establish current strength. Deleted bots are excluded.", "",
  "| Bot | Priority | Public wins/games | Win rate | Train/holdout games | Mode-derived target word length | Holdout winning-bid MAE, coins | Losing-bound violations |",
  "|---|---|---:|---:|---:|---:|---:|---:|", ...reportRows, "",
  `Audited ${audits.length} distinct completed matches; ${audits.filter((audit)=>audit.valid).length} passed exact score/balance reconstruction checks. Excluded ${audits.filter((audit)=>!audit.valid).length} inconsistent histories and ${skippedLateBalance} winning-bid observations whose amounts exceeded the balance visible at auction opening.`, "",
  "## What the models capture", "",
  "Each runnable local sample bot predicts affordable bids from letter, balance, public rack size, retained public copies, match progress, score, leader gap, and auction payment type. The learner uses exact archived winning bids, upper bounds for losses, and lower bounds where winning-bid archives are missing. Losing bids are never fabricated as zero.", "",
  "The model family is chosen between a censored linear model and per-letter median bid fractions using an internal chronological calibration split. The reported holdout consists of newer whole matches kept separate from fitting and family selection. Features are reconstructed at auction opening, without using later word plays to reveal hidden letters.", "",
  "The local word policy aims for each teacher's most frequent training word length, while retaining shorter-word fallbacks. This is a mode-derived heuristic, not a claim that the teacher never plays shorter words or uses that threshold internally. Actual local word choice uses the full shared dictionary and the sample bot's own rack. Cash-out, endgame flushing, and rack-limit behavior remain explicit heuristic fallbacks, because public history cannot reveal which unplayed words were available to the original bot.", "",
  "Several models violate many losing-bid bounds on the holdout games; they are marked limited fidelity in their model files. Winning-bid prediction is better than the per-letter median baseline, but that does not repair the missing private-state information or validate their strength as opponents.", "",
  "## Limits of validation", "",
  "MAE measures absolute prediction error on observed winning bids only. Losing-bound violations measure incompatible predictions where a rival's winning bid supplies a usable upper bound. These do not measure accuracy on all bids. Initial letters, losing bids, latency, code changes within a version, and private strategy remain unknown. These are approximate opponents, not exact replicas; their original win percentages must not be attributed to the local models. No local match-strength or online performance claim has been validated here.", "",
  "## Files and use", "",
  "Model files: `bot-starter/opponents/models/bot-ID.json`. Reusable interface: `createSampleBot(model)` from `bot-starter/src/public-opponents.ts`, providing `decideBid(ctx)` and `chooseWord(ctx)` for a future local arena. It has no networking or registration behavior.", "",
  "Run `npm.cmd run opponents:download` from bot-starter to refresh the anonymous read-only snapshot, then `npm.cmd run opponents:fit` to rebuild and validate models. Raw data is ignored by Git; summary.json records query provenance and SHA-256 hashes.", "",
  `Source endpoint: ${manifest.endpoint}`, `Snapshot: ${directory}`, "",
];
writeFileSync(resolve("../PUBLIC-OPPONENTS.md"),report.join("\n"));
console.log(`Built ${models.length} local opponent models. Audit: ${summary.audit.valid}/${summary.audit.matches} matches valid.`);
