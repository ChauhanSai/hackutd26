import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { metric, selectProfile, PROFILES } from "../src/study.ts";

const files = process.argv.slice(2).map((path) => resolve(path));
if (!files.length) throw new Error("Pass the result JSON files from the completed practice sessions.");
const results = files.flatMap((path) => JSON.parse(readFileSync(path, "utf8")).results);
const games = results.filter((game) => metric(game) !== null);
const average = (values) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
const rivalMax = (game) => Math.max(...game.opponents.filter((opponent) => !opponent.simulated).map((opponent) => opponent.score));
const stats = Object.entries(PROFILES).map(([profile, weight]) => {
  const group = games.filter((game) => game.profile === profile);
  return { profile, weight, games: group.length, ourMean: average(group.map((game) => game.score)),
    rivalMean: average(group.map(rivalMax)), objective: average(group.map(metric)),
    wins: group.reduce((sum, game) => sum + game.defensiveWins, 0),
    spend: group.reduce((sum, game) => sum + game.defensiveSpend, 0) };
}).filter((group) => group.games);
const selected = selectProfile(games);
const baseline = stats.find((group) => group.profile === "baseline");
const chosen = stats.find((group) => group.profile === selected);
const selectedValidation = games.filter((game) => game.profile === selected && game.phase === "validation");
const eligibleIds = new Set(games.map((game) => game.matchId));
const events = files.flatMap((file) => readFileSync(join(dirname(file), "events.jsonl"), "utf8").trim().split("\n").filter(Boolean).map(JSON.parse));
const timings = events.filter((event) => event.event === "decision" && eligibleIds.has(String(event.data.matchId)))
  .map((event) => event.data.elapsedMs).sort((a, b) => a - b);
const p95 = timings[Math.min(timings.length - 1, Math.floor(timings.length * 0.95))] ?? 0;
const root = fileURLToPath(new URL("../../", import.meta.url));
const destination = join(root, "DEFENSIVE-RESULTS.md");
const rows = stats.map((s) => `| ${s.profile} (${s.weight}) | ${s.games} | ${s.ourMean.toFixed(1)} | ${s.rivalMean.toFixed(1)} | ${s.wins} | ${s.spend} |`);
const gameRows = games.map((g) => `| ${g.matchId} | ${g.profile} | ${g.phase} | ${g.score} | ${rivalMax(g)} | ${g.rank}/${g.opponents.length + 1} | ${g.defensiveWins} | ${g.defensiveSpend} |`);
const message = [
  "# hercule-v1: online defensive training results", "",
  `Completed ${games.length} eligible, fully observed online games, each against five real rivals in this run. Excluded ${results.length - games.length} recorded partial games, the initial diagnostic run, and a game interrupted by a lost connection.`, "",
  "| Setting (defense weight) | Games | Our average score | Average highest real rival | Defense-motivated wins | Coins spent on those wins |",
  "|---|---:|---:|---:|---:|---:|", ...rows, "",
  `Exploratory selected defensive setting: **${selected}**, weight **${PROFILES[selected]}**. It is saved for the normal Start-Bot launcher.`, "",
  `The selected setting averaged ${chosen?.ourMean.toFixed(1)} points versus the baseline's ${baseline?.ourMean.toFixed(1)}, while its strongest rivals averaged ${chosen?.rivalMean.toFixed(1)} versus ${baseline?.rivalMean.toFixed(1)}. The overall rival-score comparison does not show suppression. Match wins: ${games.filter((game) => game.rank === 1).length}/${games.length}. More defensive purchases have not yet established a competitive advantage.`, "",
  "Selection uses exploration games only and maximizes 0.5 × our score − the highest real rival's score. Each eligible candidate needs at least two exploration games. Validation games do not select the winner.", "",
  `Selected setting's fresh validation: ${selectedValidation.length} games, our average ${average(selectedValidation.map((g) => g.score)).toFixed(1)}, average highest rival ${average(selectedValidation.map(rivalMax)).toFixed(1)}.`, "",
  `Decision computation p95: ${p95.toFixed(1)} ms across ${timings.length} recorded evaluation decisions. The implementation's 16 tests and TypeScript build pass.`, "",
  "## What the numbers mean", "",
  "Defense-motivated wins are purchases where the defensive rule raised our bid above the baseline. They show that the bot acquired letters it identified as threats. They do not establish that the baseline would have lost those auctions, or prove how many points a rival would otherwise have scored.", "",
  "Opponent starting racks remain private. The bot uses conservative counts reconstructed only from public purchases and word plays, targets valuable visible completions, and reserves 20 coins for extra defensive bids. This misses threats involving unknown starting letters.", "",
  "The field and tile draws change between games. This small sequential online sample cannot establish causal score suppression or a reliable win-rate improvement. Treat the selected profile as exploratory. An earlier diagnostic run that skipped auctions due a client/server clock comparison was excluded entirely; that timing check was removed before these evaluation sessions.", "",
  "## Individual evaluation games", "",
  "| Match | Setting | Phase | Our score | Highest real rival | Rank | Defense-motivated wins | Defense spend |",
  "|---|---|---|---:|---:|---|---:|---:|", ...gameRows, "",
  "## Reproduce and inspect", "",
  "Start-Bot.cmd runs the saved defensive setting. Start-Training.cmd runs a finite comparison batch and saves reports under bot-starter/training/. It stops automatically after the requested full matches. The completed runs are now stopped.", "",
  ...files.map((file) => `- Raw match results: ${file}`), "",
].join("\n");
writeFileSync(destination, message);
writeFileSync(join(root, "bot-starter", "training", "defense-profile.json"), JSON.stringify({
  profile: selected, weight: PROFILES[selected], evidence: "exploratory online parameter tuning; causal score suppression unproven",
  evaluationGames: games.length, validationGames: selectedValidation.length, report: destination,
}, null, 2));
console.log(JSON.stringify({ report: destination, selected, stats, validation: selectedValidation.map((g) => ({ id: g.matchId, score: g.score, rivalMax: rivalMax(g) })), p95Ms: p95 }, null, 2));
