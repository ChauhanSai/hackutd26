// Anonymous SELECT requests only. This script never creates a bot or joins a lobby.
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const endpoint = "https://maincloud.spacetimedb.com/v1/database/scrabblebot/sql";
const collectedAt = new Date().toISOString();
const directory = resolve("opponents/data", collectedAt.replace(/[:.]/g, "-"));
mkdirSync(directory, { recursive: true });
const queries = [];
function decode(payload) {
  const table = payload[0];
  if (!table?.schema?.elements || !Array.isArray(table.rows)) throw new Error("Unexpected public SQL response.");
  const keys = table.schema.elements.map((field) => field.name.some);
  return table.rows.map((row) => Object.fromEntries(keys.map((key, i) => [key, row[i]])));
}
async function sql(query, filename) {
  if (!query.startsWith("SELECT ")) throw new Error("Only public reads are supported.");
  let payload;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "text/plain" },
        body: query, signal: AbortSignal.timeout(60000) });
      if (!response.ok) throw new Error(`Public SQL ${response.status}: ${(await response.text()).slice(0,200)}`);
      payload = await response.json();
      break;
    } catch (error) {
      if (attempt === 2) throw error;
      await new Promise((done) => setTimeout(done, 1000 * (attempt + 1)));
    }
  }
  writeFileSync(join(directory, filename), JSON.stringify(payload));
  const rows = decode(payload);
  queries.push({ query, filename, rows: rows.length, collectedAt: new Date().toISOString() });
  console.log(`${filename}: ${rows.length} public rows`);
  return rows;
}

const [bots, stats, versions, retired, matches] = await Promise.all([
  sql("SELECT * FROM bot", "bot.json"), sql("SELECT * FROM bot_stats", "bot_stats.json"),
  sql("SELECT * FROM bot_version", "bot_version.json"), sql("SELECT * FROM retired_bot_stats", "retired_bot_stats.json"),
  sql("SELECT * FROM match_state", "match_state.json"),
]);
const botById = new Map(bots.map((bot) => [bot.id, bot]));
const leaderboard = stats.filter((stat) => botById.has(stat.bot_id) && !botById.get(stat.bot_id).is_simulated && stat.matches_played > 0)
  .map((stat) => ({ botId: stat.bot_id, name: botById.get(stat.bot_id).name, matches: stat.matches_played, wins: stat.wins,
    winRate: stat.wins / stat.matches_played, rating: stat.rating,
    version: Math.max(1, ...versions.filter((v) => v.bot_id === stat.bot_id).map((v) => v.version)),
    versionStartedAt: Math.max(0, ...versions.filter((v) => v.bot_id === stat.bot_id).map((v) => Number(v.started_at[0]))),
  })).sort((a,b) => b.winRate-a.winRate || b.matches-a.matches);
const priority = leaderboard.filter((bot) => bot.matches >= 50).slice(0,5);
const exploratory = leaderboard.filter((bot) => bot.matches >= 10 && bot.matches < 50).slice(0,2);
const selected = [...priority.map((bot) => ({ ...bot, tier: "priority" })), ...exploratory.map((bot) => ({ ...bot, tier: "small-sample" }))];
const matchById = new Map(matches.filter((match) => Object.hasOwn(match.status, "ended")).map((match) => [match.id, match]));
// Sum values may be encoded as [variantIndex, payload] rather than objects.
if (!matchById.size) for (const match of matches) {
  if (Array.isArray(match.status) && match.status[0] === 2) matchById.set(match.id, match);
}
const selectedMatchIds = new Set();
for (const bot of selected) {
  const participation = await sql(`SELECT * FROM match_participant WHERE bot_id = ${bot.botId}`, `participants-bot-${bot.botId}.json`);
  bot.matchIds = participation.map((row) => matchById.get(row.match_id)).filter((match) => match &&
    Number(match.created_at[0]) >= bot.versionStartedAt).sort((a,b) => b.id-a.id).slice(0,80).map((match) => match.id);
  bot.matchIds.forEach((id) => selectedMatchIds.add(id));
}
if (!selectedMatchIds.size) throw new Error("No completed matches found for selected bots; check match-status encoding.");
const minimum = Math.min(...selectedMatchIds);
const maximum = Math.max(...selectedMatchIds);
const where = `WHERE match_id >= ${minimum} AND match_id <= ${maximum}`;
await Promise.all([
  sql(`SELECT * FROM auction_result ${where}`, "auction_result.json"),
  sql(`SELECT * FROM auction_top_bid_archive ${where}`, "auction_top_bid_archive.json"),
  sql(`SELECT * FROM word_play ${where}`, "word_play.json"),
  sql(`SELECT * FROM match_participant ${where}`, "match_participant.json"),
]);
const manifest = { collectedAt, completedAt: new Date().toISOString(), endpoint, readOnly: true, anonymous: true,
  definition: "Current-version wins / matches_played; tied first-place scores count as wins. Deleted bot entries excluded.",
  selection: "Top five current win rates with at least 50 matches, plus top two with 10–49 matches labeled small-sample.",
  leaderboard, selected, selectedMatchIds: [...selectedMatchIds].sort((a,b) => a-b), queries,
  retiredVersionsAvailable: retired.length,
};
writeFileSync(join(directory,"manifest.json"),JSON.stringify(manifest,null,2));
writeFileSync("opponents/latest-snapshot.json",JSON.stringify({ directory },null,2));
console.log(JSON.stringify({ directory, selected: selected.map(({name,matches,wins,winRate,tier,matchIds}) =>
  ({name,matches,wins,winRate,tier,historyMatches:matchIds.length})) },null,2));
