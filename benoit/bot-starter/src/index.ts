// ScrabbleBot bot starter (Benoit).
//
// Fast, robust online queueing and real-time ONNX bidding/play model.

import {
  DbConnection,
  type EventContext,
  type ErrorContext,
  type SubscriptionHandle,
} from "./module_bindings/index.js";
import { Identity } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { chooseWord, decideBidAsync } from "./strategy.js";
import { benoitModel } from "./model_inference.js";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "benoit";
const BOT_TOKEN = process.env.BOT_TOKEN; // from the website, first run only
const BOT_NONCE = process.env.BOT_NONCE; // from the CLI, first run only
const TOURNAMENT_ID = process.env.TOURNAMENT_ID;

const tokenCandidates = [
  path.join(process.cwd(), `.token-${BOT_NAME}`),
  path.join(currentDir, "..", `.token-${BOT_NAME}`),
  path.join(process.cwd(), ".token-benoit"),
  path.join(currentDir, "..", ".token-benoit"),
];

function loadToken(): string | undefined {
  for (const candidate of tokenCandidates) {
    try {
      if (fs.existsSync(candidate)) {
        const tok = fs.readFileSync(candidate, "utf8").trim();
        if (tok) return tok;
      }
    } catch {
      // ignore
    }
  }
  return undefined;
}

function saveToken(tok: string) {
  const savePaths = [
    path.join(process.cwd(), `.token-${BOT_NAME}`),
    path.join(currentDir, "..", `.token-${BOT_NAME}`),
  ];
  for (const p of savePaths) {
    try {
      fs.writeFileSync(p, tok);
    } catch {
      // ignore
    }
  }
}

// Load the shared wordlist so the bot can pick playable words locally.
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
let activeMatchId: bigint | null = null;
let matchSub: SubscriptionHandle | null = null;
const finishedMatches = new Set<string>();
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
  const key = `${matchId}:${auctionId}`;
  if (bidsByAuction.has(key)) return;
  bidsByAuction.add(key);

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
  const modelTag = benoitModel.isReady ? "NN" : "heuristic";
  console.log(
    `[${BOT_NAME}] bid ${amount} on '${letter}' (match ${matchId}, auction ${auctionId}) [${modelTag}]`,
  );
}

function tryPlayWord(conn: DbConnection, matchId = activeMatchId) {
  if (matchId === null) return;
  const m = conn.db.matchState.id.find(matchId);
  if (m?.status.tag !== "Running") return;

  const key = String(matchId);
  const now = Date.now();
  if (now - (lastWordAttemptByMatch.get(key) ?? 0) < WORD_RETRY_MS) return;
  lastWordAttemptByMatch.set(key, now);

  const participant = participantForMatch(conn, matchId);
  const word = chooseWord({
    myRack: rackForMatch(conn, matchId),
    dictionary: DICTIONARY,
    myBalance: participant?.balance ?? 100,
  });
  if (!word) return;
  console.log(`[${BOT_NAME}] match ${matchId}: playing '${word}'`);
  conn.reducers
    .submitWord({ matchId, word })
    .catch(logFailure(`playing '${word}'`));
}

function attachMatch(conn: DbConnection, matchId: bigint) {
  if (activeMatchId === matchId || finishedMatches.has(String(matchId))) return;
  const m = conn.db.matchState.id.find(matchId);
  if (m?.status.tag !== "Running") return;

  activeMatchId = matchId;
  console.log(`[${BOT_NAME}] *** ENTERED MATCH ${matchId} ***`);

  matchSub?.unsubscribe();
  matchSub = conn
    .subscriptionBuilder()
    .onError((ctx) => {
      console.warn(`[${BOT_NAME}] match ${matchId} subscription error: ${ctx.event}`);
    })
    .onApplied(() => {
      console.log(`[${BOT_NAME}] match ${matchId} subscription live`);
      for (const a of conn.db.auction.iter()) {
        if (a.matchId === matchId && a.status.tag === "Open") {
          tryBid(conn, a.id, matchId, a.letter);
        }
      }
      tryPlayWord(conn, matchId);
    })
    .subscribe([
      `SELECT * FROM match_state WHERE id = ${matchId}`,
      `SELECT * FROM match_participant WHERE match_id = ${matchId}`,
      `SELECT * FROM auction WHERE match_id = ${matchId}`,
      `SELECT * FROM auction_result WHERE match_id = ${matchId}`,
      `SELECT * FROM word_play WHERE match_id = ${matchId}`,
      `SELECT * FROM my_rack WHERE match_id = ${matchId}`,
    ]);
}

function attachMatches(conn: DbConnection) {
  if (myBotId === null) return;
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.botId !== myBotId) continue;
    const matchId = p.matchId;
    const key = String(matchId);
    if (finishedMatches.has(key) || activeMatchId === matchId) continue;
    const m = conn.db.matchState.id.find(matchId);
    if (m?.status.tag !== "Running") continue;
    attachMatch(conn, matchId);
    break;
  }
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

let joinPending = false;
function joinLobby(conn: DbConnection) {
  if (myBotId === null || activeMatchId !== null || joinPending) return;

  const alreadyIn = Array.from(conn.db.openLobbyMembers.iter()).some(
    (lm) => lm.botId === myBotId,
  );
  if (alreadyIn) return;

  joinPending = true;
  console.log(`[${BOT_NAME}] joining lobby`);
  conn.reducers
    .joinLobby({})
    .then(() => {
      console.log(`[${BOT_NAME}] joined lobby queue`);
    })
    .catch((err) => {
      const msg = err instanceof Error ? err.message : String(err);
      if (!msg.includes("already in a running match")) {
        console.warn(`[${BOT_NAME}] joining the lobby failed: ${msg}`);
      }
    })
    .finally(() => {
      joinPending = false;
    });
}

function onConnect(conn: DbConnection, identity: Identity, token: string) {
  myIdentity = identity;
  saveToken(token);
  console.log(`[${BOT_NAME}] connected as ${identity.toHexString()}`);

  // 1. Subscribe to small account tables to resolve our bot identity
  conn
    .subscriptionBuilder()
    .onError((ctx) => {
      console.error(`[${BOT_NAME}] account subscription error: ${ctx.event}`);
    })
    .onApplied(() => {
      myBotId = resolveMyBotId(conn);
      if (myBotId === null) {
        if (BOT_NONCE) {
          console.log(`[${BOT_NAME}] claiming credential with nonce…`);
          conn.reducers
            .claimCredential({ code: BOT_NONCE })
            .catch(logFailure("claiming the credential"));
          setTimeout(() => {
            myBotId = resolveMyBotId(conn);
            if (myBotId === null) {
              console.error(`[${BOT_NAME}] couldn't claim credential. Bad / expired nonce?`);
              process.exit(1);
            }
            startBotOperations(conn);
          }, 1000);
          return;
        } else {
          console.error(
            `[${BOT_NAME}] no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.`,
          );
          process.exit(1);
        }
      }

      startBotOperations(conn);
    })
    .subscribe(["SELECT * FROM bot_credential", "SELECT * FROM bot"]);

  // Table listeners for active match events
  conn.db.matchParticipant.onInsert(() => {
    setImmediate(() => attachMatches(conn));
  });
  conn.db.matchState.onInsert(() => {
    setImmediate(() => attachMatches(conn));
  });

  conn.db.auction.onInsert((_ctx: EventContext, a) => {
    if (a.matchId === activeMatchId && a.status.tag === "Open") {
      tryBid(conn, a.id, a.matchId, a.letter);
    }
  });

  conn.db.myRack.onInsert(() => {
    if (activeMatchId !== null) tryPlayWord(conn, activeMatchId);
  });
  conn.db.myRack.onUpdate(() => {
    if (activeMatchId !== null) tryPlayWord(conn, activeMatchId);
  });

  conn.db.auctionResult.onInsert((_ctx, r) => {
    if (myBotId === null || r.matchId !== activeMatchId) return;
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
    if (wp.matchId !== activeMatchId) return;
    benoitModel.eventBuffer.recordWordPlay(Number(wp.botId), wp.word, Number(wp.totalReward));
  });

  // When match ends, log summary and immediately re-queue
  conn.db.matchState.onUpdate((_ctx, oldM, newM) => {
    if (newM.id === activeMatchId && newM.status.tag === "Ended") {
      finishedMatches.add(String(newM.id));
      logMatchSummary(conn, newM.id);
      benoitModel.eventBuffer.clear();
      matchSub?.unsubscribe();
      matchSub = null;
      activeMatchId = null;
      console.log(`[${BOT_NAME}] match ${newM.id} ended; rejoining lobby`);
      setTimeout(() => joinLobby(conn), 250);
    } else {
      attachMatches(conn);
    }
  });
}

function startBotOperations(conn: DbConnection) {
  const bot = conn.db.bot.id.find(myBotId!);
  console.log(`[${BOT_NAME}] acting as bot '${bot?.name ?? "?"}' (id ${myBotId})`);

  if (TOURNAMENT_ID) {
    conn.reducers
      .registerForTournament({ tournamentId: BigInt(TOURNAMENT_ID) })
      .then(() => console.log(`[${BOT_NAME}] registered for tournament ${TOURNAMENT_ID}`))
      .catch(logFailure(`registering for tournament ${TOURNAMENT_ID}`));
  }

  // 2. Subscribe to scoped lobby and participant queries
  conn
    .subscriptionBuilder()
    .onError((ctx) => {
      console.warn(`[${BOT_NAME}] lobby subscription error: ${ctx.event}`);
    })
    .onApplied(() => {
      console.log(`[${BOT_NAME}] lobby queue listener ready`);

      // Check if already in a running match
      attachMatches(conn);

      // Initial join
      joinLobby(conn);

      // Heartbeat queue loop: ensures bot is always queuing when not in a match
      const heartbeat = setInterval(() => {
        attachMatches(conn);
        if (activeMatchId === null) {
          joinLobby(conn);
        }
      }, 3000);
      heartbeat.unref();
    })
    .subscribe([
      `SELECT match_state.* FROM match_state JOIN match_participant ON match_state.id = match_participant.match_id WHERE match_participant.bot_id = ${myBotId}`,
      `SELECT * FROM match_participant WHERE bot_id = ${myBotId}`,
      "SELECT * FROM open_lobbies",
      "SELECT * FROM open_lobby_members",
    ]);
}

function main() {
  const savedToken = BOT_TOKEN ?? loadToken();
  if (!savedToken && !BOT_NONCE) {
    console.error("No bot token saved yet. Set BOT_TOKEN or run with BOT_NONCE.");
    process.exitCode = 1;
    return;
  }
  console.log(`[${BOT_NAME}] connecting to ${DB_NAME} at ${HOST}`);
  DbConnection.builder()
    .withUri(HOST)
    .withDatabaseName(DB_NAME)
    .withToken(savedToken)
    .onConnect(onConnect)
    .onConnectError((_ctx: ErrorContext, err: Error) =>
      console.error("connect error:", err.message),
    )
    .onDisconnect(() => console.log("disconnected"))
    .build();
}

main();
