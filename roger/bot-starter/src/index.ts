// ScrabbleBot bot starter.
//
// First run, with a token from the website:
//   On your team's page, click "Generate token", then run
//   `BOT_NAME=<name> BOT_TOKEN=<token> npm start`.
//
// Or with a nonce from the CLI:
//   1. `spacetime call scrabblebot mint_credential_nonce`
//      `spacetime sql 'SELECT code FROM my_nonces ORDER BY expires_at DESC'`
//   2. `BOT_NAME=<name> BOT_NONCE=<code> npm start`
//      The bot connects fresh and redeems the nonce.
//
// Either way the bot saves its token to .token-<BOT_NAME>, so later runs
// need only `BOT_NAME=<name> npm start`.
//
// Entering a tournament: `BOT_NAME=<name> TOURNAMENT_ID=<id> npm start`
// registers the bot for that tournament when it connects. Only the bot
// itself can register, so this has to come from the bot's process.
//
// Edit ./src/strategy.ts to customise how your bot bids and plays words.

import {
  DbConnection,
  type EventContext,
  type ErrorContext,
} from "./module_bindings/index.js";
import { Identity } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { chooseWord, decideBid, decideBidAsync } from "./strategy.js";
import { benoitModel } from "./model_inference.js";

const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "bot";
const BOT_TOKEN = process.env.BOT_TOKEN; // from the website, first run only
const BOT_NONCE = process.env.BOT_NONCE; // from the CLI, first run only
const TOURNAMENT_ID = process.env.TOURNAMENT_ID;
const TOKEN_PATH = path.join(process.cwd(), `.token-${BOT_NAME}`);

function loadToken(): string | undefined {
  try {
    return fs.readFileSync(TOKEN_PATH, "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}
function saveToken(tok: string) {
  fs.writeFileSync(TOKEN_PATH, tok);
}

// Load the shared wordlist so the bot can pick playable words locally.
const currentDir = path.dirname(fileURLToPath(import.meta.url));
const candidatePaths = [
  path.join(currentDir, "..", "..", "spacetimedb", "wordlist.txt"),
  path.join(currentDir, "..", "spacetimedb", "wordlist.txt"),
  path.join(process.cwd(), "spacetimedb", "wordlist.txt"),
  path.join(process.cwd(), "..", "spacetimedb", "wordlist.txt"),
  "/Users/saichauhan/Desktop/HackUTD '26/scrabblebot/spacetimedb/wordlist.txt",
];
const dictionaryPath = candidatePaths.find((p) => fs.existsSync(p)) ?? candidatePaths[0]!;
const DICTIONARY: string[] = fs.existsSync(dictionaryPath)
  ? fs
      .readFileSync(dictionaryPath, "utf8")
      .split("\n")
      .map((l) => l.trim().toUpperCase())
      .filter((l) => l.length >= 2)
  : [];
DICTIONARY.sort((a, b) => b.length - a.length); // longest first for greedy pick
console.log(`[${BOT_NAME}] loaded ${DICTIONARY.length} words from dictionary`);

let myIdentity: Identity | null = null;
let myBotId: bigint | null = null;
const bidsByAuction = new Set<string>();
const lastWordAttemptByMatch = new Map<string, number>();
const WORD_RETRY_MS = 500;

function resolveMyBotId(conn: DbConnection): bigint | null {
  if (!myIdentity) return null;
  const cred = conn.db.botCredential.identity.find(myIdentity);
  return cred ? cred.botId : null;
}

function rackForMatch(
  conn: DbConnection,
  matchId: bigint,
): Map<string, number> {
  const rack = new Map<string, number>();
  for (const h of conn.db.myRack.iter()) {
    if (h.matchId !== matchId) continue;
    rack.set(h.letter, (rack.get(h.letter) ?? 0) + h.count);
  }
  return rack;
}

function participantForMatch(
  conn: DbConnection,
  matchId: bigint,
): { balance: number; score: number } | null {
  if (myBotId === null) return null;
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.matchId !== matchId) continue;
    if (p.botId !== myBotId) continue;
    return { balance: Number(p.balance), score: Number(p.score) };
  }
  return null;
}

// A reducer call fails when the module refuses it, often for an expected
// reason: a word played just after its match ended, a bid that missed the
// auction's window. Log it and keep playing; an uncaught failure would stop
// the bot.
function logFailure(what: string) {
  return (err: unknown) =>
    console.warn(
      `[${BOT_NAME}] ${what} failed: ${err instanceof Error ? err.message : err}`,
    );
}

async function tryBid(
  conn: DbConnection,
  auctionId: bigint,
  matchId: bigint,
  letter: string,
) {
  if (myBotId === null) return;
  const key = `${matchId}:${auctionId}`;
  if (bidsByAuction.has(key)) return;
  const participant = participantForMatch(conn, matchId);
  if (!participant) return;
  const rack = rackForMatch(conn, matchId);
  const matchState = conn.db.matchState.id.find(matchId);
  const roundNum = matchState?.currentRound ?? 1;

  const amount = await decideBidAsync({
    letter,
    myBalance: participant.balance,
    myRack: rack,
    roundNum,
  });
  if (amount <= 0) return;
  conn.reducers
    .submitBid({ auctionId, amount: BigInt(amount) })
    .catch(logFailure(`bid on auction ${auctionId}`));
  bidsByAuction.add(key);
  const modelTag = benoitModel.isReady ? "NN" : "heuristic";
  console.log(
    `[${BOT_NAME}] bid ${amount} on '${letter}' (match ${matchId}, auction ${auctionId}) [${modelTag}]`,
  );
}

function tryPlayWord(conn: DbConnection) {
  if (myBotId === null) return;
  // Only matches still being played. The bot keeps its letters from
  // matches that have ended, but can't play them.
  const matches = new Set<bigint>();
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.botId !== myBotId) continue;
    const m = conn.db.matchState.id.find(p.matchId);
    if (m?.status.tag === "Running") matches.add(p.matchId);
  }
  const now = Date.now();
  for (const matchId of matches) {
    const key = String(matchId);
    if (now - (lastWordAttemptByMatch.get(key) ?? 0) < WORD_RETRY_MS) continue;
    lastWordAttemptByMatch.set(key, now);
    const participant = participantForMatch(conn, matchId);
    const word = chooseWord({
      myRack: rackForMatch(conn, matchId),
      dictionary: DICTIONARY,
      myBalance: participant?.balance ?? 100,
    });
    if (!word) continue;
    console.log(`[${BOT_NAME}] match ${matchId}: playing '${word}'`);
    conn.reducers
      .submitWord({ matchId, word })
      .catch(logFailure(`playing '${word}'`));
  }
}

function onConnect(conn: DbConnection, identity: Identity, token: string) {
  myIdentity = identity;
  saveToken(token);
  console.log(`[${BOT_NAME}] connected as ${identity.toHexString()}`);

  conn
    .subscriptionBuilder()
    .onApplied(() => {
      console.log(`[${BOT_NAME}] subscription applied`);

      // Resolve our bot id, possibly after claiming a nonce.
      myBotId = resolveMyBotId(conn);
      if (myBotId === null) {
        if (BOT_NONCE) {
          console.log(`[${BOT_NAME}] claiming credential with nonce…`);
          conn.reducers
            .claimCredential({ code: BOT_NONCE })
            .catch(logFailure("claiming the credential"));
          // Wait briefly for the credential to land; check again.
          setTimeout(() => {
            myBotId = resolveMyBotId(conn);
            if (myBotId === null) {
              console.error(
                `[${BOT_NAME}] couldn't claim credential. Bad / expired nonce?`,
              );
              process.exit(1);
            }
            const bot = conn.db.bot.id.find(myBotId);
            console.log(
              `[${BOT_NAME}] claimed credential for bot '${bot?.name ?? "?"}' (id ${myBotId})`,
            );
            bootstrapActivity(conn);
          }, 1000);
          return;
        } else {
          console.error(
            `[${BOT_NAME}] no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.`,
          );
          process.exit(1);
        }
      }

      const bot = conn.db.bot.id.find(myBotId);
      console.log(
        `[${BOT_NAME}] acting as bot '${bot?.name ?? "?"}' (id ${myBotId})`,
      );
      bootstrapActivity(conn);
    })
    .subscribeToAllTables();

  conn.db.auction.onInsert((_ctx: EventContext, a) => {
    if (a.status.tag === "Open") tryBid(conn, a.id, a.matchId, a.letter);
  });
  conn.db.myRack.onInsert(() => tryPlayWord(conn));
  conn.db.myRack.onUpdate(() => tryPlayWord(conn));
  conn.db.botCredential.onInsert(() => {
    if (myBotId === null) myBotId = resolveMyBotId(conn);
  });
  conn.db.auctionResult.onInsert((_ctx, r) => {
    if (myBotId === null) return;
    const winnerIdx = r.winnerBotId !== undefined && r.winnerBotId !== null ? Number(r.winnerBotId) : null;
    benoitModel.eventBuffer.recordAuctionResult(winnerIdx, r.letter, Number(r.paid));
    const winner =
      r.winnerBotId !== undefined && r.winnerBotId !== null
        ? String(r.winnerBotId)
        : "no-bid";
    console.log(
      `[${BOT_NAME}] match ${r.matchId} auction ${r.auctionId} '${r.letter}' → bot ${winner} paid ${r.paid}`,
    );
  });

  conn.db.wordPlay.onInsert((_ctx, wp) => {
    benoitModel.eventBuffer.recordWordPlay(Number(wp.botId), wp.word, Number(wp.totalReward));
  });

  // When a match this bot was in ends, hop back into the lobby.
  conn.db.matchState.onUpdate((_ctx, old, neu) => {
    if (myBotId === null) return;
    if (old.status.tag !== "Ended" && neu.status.tag === "Ended") {
      benoitModel.eventBuffer.clear();
      const wasIn = Array.from(conn.db.matchParticipant.iter()).some(
        (p) => p.matchId === neu.id && p.botId === myBotId,
      );
      if (wasIn) {
        logMatchSummary(conn, neu.id);
        console.log(`[${BOT_NAME}] match ${neu.id} ended; rejoining lobby`);
        joinLobby(conn);
      }
    }
  });
}

function logMatchSummary(conn: DbConnection, matchId: bigint) {
  const participants = Array.from(conn.db.matchParticipant.iter())
    .filter((p) => p.matchId === matchId)
    .sort((a, b) => Number(b.score - a.score) || Number(b.balance - a.balance));

  if (participants.length === 0) return;

  const matchState = conn.db.matchState.id.find(matchId);
  const startingRackSize = matchState?.startingRackSize ?? 5;

  const results = Array.from(conn.db.auctionResult.iter()).filter((r) => r.matchId === matchId);
  const plays = Array.from(conn.db.wordPlay.iter()).filter((w) => w.matchId === matchId);

  console.log(`\n==================== MATCH ${matchId} FINAL RESULTS ====================`);
  console.log(`Rank | ${"Player".padEnd(26)} | ${"Score".padStart(5)} | ${"Coins".padStart(5)} | Letters Remaining`);
  console.log("-".repeat(78));

  participants.forEach((p, idx) => {
    const isMe = myBotId !== null && p.botId === myBotId;
    const bot = conn.db.bot.id.find(p.botId);
    const botName = bot?.name ?? `bot ${p.botId}`;
    const displayName = `${botName} (${p.botId})${isMe ? " [YOU]" : ""}`.padEnd(26);
    const scoreStr = String(p.score).padStart(5);
    const coinsStr = String(p.balance).padStart(5);
    const rankStr = String(idx + 1).padStart(4);

    let lettersSummary = "";
    if (isMe) {
      const myLetters: string[] = [];
      let totalCount = 0;
      for (const h of conn.db.myRack.iter()) {
        if (h.matchId !== matchId) continue;
        for (let i = 0; i < h.count; i++) {
          myLetters.push(h.letter);
          totalCount++;
        }
      }
      myLetters.sort();
      lettersSummary =
        totalCount === 0
          ? "0 tiles"
          : `${totalCount} tile${totalCount === 1 ? "" : "s"} (${myLetters.join(", ")})`;
    } else {
      const known = new Map<string, number>();
      let hidden = startingRackSize;

      for (const r of results) {
        if (r.winnerBotId === p.botId) {
          known.set(r.letter, (known.get(r.letter) ?? 0) + 1);
        }
      }

      for (const wp of plays) {
        if (wp.botId === p.botId) {
          for (const c of wp.word) {
            const have = known.get(c) ?? 0;
            if (have > 0) {
              known.set(c, have - 1);
            } else {
              hidden = Math.max(0, hidden - 1);
            }
          }
        }
      }

      const knownList: string[] = [];
      let knownCount = 0;
      for (const [letter, count] of known.entries()) {
        for (let i = 0; i < count; i++) {
          knownList.push(letter);
          knownCount++;
        }
      }
      knownList.sort();

      const totalCount = knownCount + hidden;
      if (totalCount === 0) {
        lettersSummary = "0 tiles";
      } else {
        const details: string[] = [];
        if (knownCount > 0) details.push(knownList.join(", "));
        if (hidden > 0) details.push(`${hidden} hidden`);
        lettersSummary = `${totalCount} tile${totalCount === 1 ? "" : "s"} (${details.join(" + ")})`;
      }
    }

    console.log(`${rankStr} | ${displayName} | ${scoreStr} | ${coinsStr} | ${lettersSummary}`);
  });

  console.log("=".repeat(78) + "\n");
}

function joinLobby(conn: DbConnection) {
  if (myBotId === null) return;
  const inRunning = Array.from(conn.db.matchParticipant.iter()).some((p) => {
    if (p.botId !== myBotId) return false;
    const m = conn.db.matchState.id.find(p.matchId);
    return m?.status.tag === "Running";
  });
  if (inRunning) return;
  const openLobby = Array.from(conn.db.lobby.iter()).find(
    (l) => l.status.tag === "Open",
  );
  const alreadyIn =
    openLobby !== undefined &&
    Array.from(conn.db.lobbyMember.iter()).some(
      (lm) => lm.lobbyId === openLobby.id && lm.botId === myBotId,
    );
  if (alreadyIn) return;
  console.log(`[${BOT_NAME}] joining lobby`);
  conn.reducers.joinLobby({}).catch(logFailure("joining the lobby"));
}

function bootstrapActivity(conn: DbConnection) {
  if (TOURNAMENT_ID) {
    conn.reducers
      .registerForTournament({ tournamentId: BigInt(TOURNAMENT_ID) })
      .then(() =>
        console.log(`[${BOT_NAME}] registered for tournament ${TOURNAMENT_ID}`),
      )
      .catch(logFailure(`registering for tournament ${TOURNAMENT_ID}`));
  }
  joinLobby(conn);
  for (const a of conn.db.auction.iter()) {
    if (a.status.tag === "Open") tryBid(conn, a.id, a.matchId, a.letter);
  }
  tryPlayWord(conn);
}

function main() {
  console.log(`[${BOT_NAME}] connecting to ${DB_NAME} at ${HOST}`);
  DbConnection.builder()
    .withUri(HOST)
    .withDatabaseName(DB_NAME)
    .withToken(BOT_TOKEN ?? loadToken())
    .onConnect(onConnect)
    .onConnectError((_ctx: ErrorContext, err: Error) =>
      console.error("connect error:", err.message),
    )
    .onDisconnect(() => console.log("disconnected"))
    .build();
}

main();
