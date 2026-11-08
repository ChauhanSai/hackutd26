import { readFileSync, writeFileSync, existsSync, unlinkSync } from "node:fs";
import { resolve, join } from "node:path";
import { DbConnection, type SubscriptionHandle } from "./module_bindings/index.js";
import { loadDictionary } from "./dictionary.js";
import { prepareDictionary } from "./strategy.js";
import { observeRacks, type DefensiveDecision, type PublicRackEvent } from "./defense.js";
import {decideHerculeBid,chooseHerculeWord,HERCULE_POLICY_VERSION,HERCULE_ACTIVE_CONFIG} from './hercule.js';
import {HERCULE_PRIOR_METADATA} from './hercule-priors.js';
import {createLiveHerculePolicy,recoverHerculeKnowledge,HERCULE_LIVE_VERSION,HERCULE_LIVE_CONFIG} from './hercule-live.js';
import { Study, PROFILES, type Profile, type MatchResult } from "./study.js";
import { decideWagsDecision, chooseWagsWord, prepareWagsDictionary, assertWagsIdentity, WAGS_SETTINGS } from "./wags.js";
import { VictorMateoPolicy, VICTOR_POLICY_VERSION, VICTOR_SETTINGS, assertVictorIdentity, type VictorMode } from "./victor.js";
import {SCORING_POLICY_VERSION} from "./scoring-v3.js";
import {onlineEvaluation,assertBotIdentity} from "./online-evaluation.js";
import {auctionProgress,ONLINE_ADAPTER_VERSION} from "./online-progress.js";

// Keep the old launcher spelling as an alias, never as a way to restore defense.
const requestedStrategy = process.env.BOT_STRATEGY ?? "hercule";
const strategy=requestedStrategy==='defensive'?'hercule':requestedStrategy;
if (strategy !== "hercule" && strategy !== "wags" && strategy !== "victor") throw new Error("Unknown bot strategy.");
const isWags = strategy === "wags";
const isVictor = strategy === "victor";
const name = process.env.BOT_NAME ?? (isVictor?"VictorMateo":isWags ? "Wags" : "hercule-v1");
if (!/^[A-Za-z0-9_-]{1,32}$/.test(name)) throw new Error("Invalid local bot name.");
if(isVictor&&name!=="VictorMateo") throw new Error("The hybrid strategy needs the local bot name VictorMateo.");
const host = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const database = process.env.STDB_DB ?? "scrabblebot";
const tokenPath = resolve(`.token-${name}`);
const token = process.env.BOT_TOKEN ?? (existsSync(tokenPath) ? readFileSync(tokenPath, "utf8").trim() : undefined);
const training = process.argv.includes("--train") || process.env.BOT_TRAIN === "1";
const evaluation=onlineEvaluation(process.env);
if(evaluation&&training)throw new Error('Fixed-policy evaluation cannot use the training schedule.');
if ((isWags||isVictor) && training) throw new Error(`${name} uses its own strategy; bounded hercule-v1 practice does not apply.`);
const limit = evaluation?.matches??Number(process.env.BOT_TRAIN_MATCHES ?? 8);
if (training && (!Number.isInteger(limit) || limit < 1 || limit > 50)) throw new Error("Training matches must be between 1 and 50.");
if (training && process.env.TOURNAMENT_ID) throw new Error("Practice training cannot register for a tournament. Clear TOURNAMENT_ID first.");
if (!token && !process.env.BOT_NONCE) throw new Error("Run Start-Bot.cmd and enter your team token first.");

const lockPath = resolve(`.bot-${name}.lock`);
if (existsSync(lockPath)) {
  const pid = Number(readFileSync(lockPath, "utf8"));
  let alive = false;
  try { process.kill(pid, 0); alive = true; } catch (error) { alive = (error as NodeJS.ErrnoException).code === "EPERM"; }
  if (alive) throw new Error(`Another ${name} process is running (PID ${pid}). Stop it before starting another.`);
  unlinkSync(lockPath);
}
writeFileSync(lockPath, String(process.pid), { flag: "wx" });
process.on("exit", () => {
  if (existsSync(lockPath) && readFileSync(lockPath, "utf8") === String(process.pid)) unlinkSync(lockPath);
});

const dictionary = loadDictionary();
prepareDictionary(dictionary);
if (isWags) prepareWagsDictionary(dictionary);
const study = new Study(training, limit, join("runs",name), { botName:name,strategy,fixedProfile:evaluation?.profile,policyVersion:isVictor?VICTOR_POLICY_VERSION:isWags?SCORING_POLICY_VERSION:HERCULE_LIVE_VERSION });
if (training && study.results.filter((result) => result.phase !== "partial").length >= limit) {
  study.write();
  console.log(`This practice session is already complete. Report: ${resolve(study.dir, "REPORT.md")}`);
  process.exit(0);
}
const normal: Profile = evaluation?.profile??(isVictor?"victor":isWags ? "wags" : "hercule");
study.log("session", { host, database, name, strategy, training, adapterVersion:ONLINE_ADAPTER_VERSION,evaluation:!!evaluation, limit: training||evaluation ? limit : null,minutes:evaluation?.minutes,
  policyVersion:isVictor?VICTOR_POLICY_VERSION:isWags?SCORING_POLICY_VERSION:HERCULE_LIVE_VERSION,profiles: isVictor||isWags?PROFILES:{hercule:0},plan:null,
  configuration:isVictor||isWags?undefined:HERCULE_LIVE_CONFIG,wordPriorSnapshot:isVictor||isWags?undefined:HERCULE_PRIOR_METADATA.collectedAt });
console.log(`[${name}] ${isVictor?"Build scoring income, defend a funded position, rebuild when needed":isWags ? "Aggressive acquisition, planned payouts, contender pressure" : training ? `Fixed offense-only practice: ${limit} full matches` : 'Productive acquisition and premium-word scoring; defense disabled'}`);
console.log(`Reports: ${resolve(study.dir)}`);
if(evaluation)study.write();

interface State {
  id: bigint;
  profile: Profile;
  phase: MatchResult["phase"];
  ready: boolean;
  ending: boolean;
  pendingWord: boolean;
  retryWordAt: number;
  decisions: Map<string, DefensiveDecision>;
  handledResults: Set<string>;
  defensiveBids: number;
  defensiveWins: number;
  defensiveSpend: number;
  totalSpend: number;
  rackRevision: number;
  lastSuccessfulRevision: number;
  victor?:VictorMateoPolicy;
  hercule?:ReturnType<typeof createLiveHerculePolicy>;
  subscription?: SubscriptionHandle;
}

const states = new Map<string, State>();
const finished = new Set<string>();
let connection: DbConnection | undefined;
let botId: bigint | undefined;
let subscribed = false;
let stopping = false;
let joinPending = false;
let stopJoining = false;
let connectedToken: string | undefined;

function stop(reason: string, code = 0) {
  if (stopping) return;
  stopping = true;
  study.log("stop", { reason });
  study.write();
  console.log(`[${name}] ${reason}. Report: ${resolve(study.dir, "REPORT.md")}`);
  connection?.disconnect();
  setTimeout(() => process.exit(code), 200);
}

function fail(what: string, error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  console.warn(`[${name}] ${what}: ${message}`);
  study.log("error", { what, message });
}

function rackFor(id: bigint) {
  const rack = new Map<string, number>();
  for (const row of connection!.db.myRack.iter()) if (row.matchId === id) rack.set(row.letter, row.count);
  return rack;
}

function opponentsFor(id: bigint) {
  const events: PublicRackEvent[] = [];
  for (const result of connection!.db.auctionResult.iter()) {
    if (result.matchId === id && result.winnerBotId !== undefined) events.push({
      time: result.closedAt.microsSinceUnixEpoch, kind: "purchase", botId: String(result.winnerBotId), text: result.letter,
    });
  }
  for (const word of connection!.db.wordPlay.iter()) if (word.matchId === id) events.push({
    time: word.playedAt.microsSinceUnixEpoch, kind: "word", botId: String(word.botId), text: word.word,
  });
  const participants=[...connection!.db.matchParticipant.iter()].filter(p=>p.matchId===id);
  const starting=connection!.db.matchState.id.find(id)?.startingRackSize??5;
  const racks = observeRacks(events,starting,participants.map(p=>String(p.botId)));
  return participants.filter((p) => p.botId !== botId).map((p) => {
    const bot = connection!.db.bot.id.find(p.botId);
    const stats=connection!.db.botStats.botId.find(p.botId);
    return { id: String(p.botId), name: bot?.name ?? String(p.botId), score: Number(p.score), balance: Number(p.balance),
      isSimulated: bot?.isSimulated ?? true, knownRack: racks.get(String(p.botId))?.knownRack ?? new Map<string, number>(),
      rackSize:racks.get(String(p.botId))?.rackSize??starting,
      matchesPlayed:stats?.matchesPlayed,wins:stats?.wins,rating:stats?.rating };
  });
}

function contextFor(id:bigint) {
  const match=connection!.db.matchState.id.find(id)!;
  const participants=[...connection!.db.matchParticipant.iter()].filter(p=>p.matchId===id);
  const participant=participants.find(p=>p.botId===botId)!;
  const progress=auctionProgress(match);
  const auction=match.currentAuctionId===undefined?undefined:connection!.db.auction.id.find(match.currentAuctionId);
  const open=auction?.status.tag==="Open"?auction:undefined;
  const commitment=open?states.get(String(id))?.decisions.get(String(open.id))?.amount:undefined;
  return {myRack:rackFor(id),dictionary,myBalance:Number(participant.balance),myScore:Number(participant.score),
    ...progress,auctionType:match.auctionType.tag,
    auctionLetter:open?.letter,committedBid:commitment};
}

function logVictorMode(state:State, previous:VictorMode|undefined) {
  if(!state.victor||state.victor.snapshot.mode===previous) return;
  const snapshot=state.victor.snapshot;
  study.log("hybrid-mode",{matchId:state.id,previous,...snapshot});
  console.log(`[${name}] ${previous} -> ${snapshot.mode}: ${snapshot.reason}`);
}

function bid(auctionId: bigint) {
  if (stopping || !connection || botId === undefined) return;
  const auction = connection.db.auction.id.find(auctionId);
  if (!auction || auction.status.tag !== "Open") return;
  const state = states.get(String(auction.matchId));
  if (!state?.ready || state.ending || state.decisions.has(String(auctionId))) return;
  const participant = [...connection.db.matchParticipant.iter()].find((p) => p.matchId === auction.matchId && p.botId === botId);
  if (!participant) return;
  const start = performance.now();
  const ctx = { ...contextFor(auction.matchId),letter:auction.letter };
  const opponents=opponentsFor(auction.matchId);
  let herculeDecision:DefensiveDecision|undefined;
  if(state.hercule){
    const match=connection.db.matchState.id.find(state.id)!;
    const purchases=[...connection.db.auctionResult.iter()].filter(a=>a.matchId===state.id&&a.auctionId<auctionId);
    const knowledge=recoverHerculeKnowledge({rack:ctx.myRack,ownId:botId,startingRackSize:match.startingRackSize,participants:opponents.length+1,
      purchases:purchases.map(a=>({letter:a.letter,winnerBotId:a.winnerBotId,at:a.closedAt.microsSinceUnixEpoch})),
      words:[...connection.db.wordPlay.iter()].filter(w=>w.matchId===state.id).map(w=>({word:w.word,botId:w.botId,at:w.playedAt.microsSinceUnixEpoch}))});
    if(knowledge){state.hercule.restore(knowledge);const amount=state.hercule.bid({...ctx,publicKnownRack:new Map(),opponents});herculeDecision={amount,baseline:amount,threat:0,weight:0,...state.hercule.lastDecision};}
    else {herculeDecision=decideHerculeBid(ctx);study.log('median-history-fallback',{matchId:state.id,auctionId,reason:'Own initial rack did not reconcile; using v4 for this auction'});}
  }
  const previous=state.victor?.snapshot.mode;
  const decision: DefensiveDecision = state.victor?state.victor.bid({...ctx,opponents}):isWags ? decideWagsDecision({...ctx,opponents})
    : herculeDecision??decideHerculeBid(ctx);
  logVictorMode(state,previous);
  const elapsedMs = performance.now() - start;
  state.decisions.set(String(auctionId), decision);
  // The client's wall clock may differ from the server's. The server decides
  // whether a bid is timely; an Open auction must not be skipped on local time.
  study.log("decision", { matchId: state.id, auctionId, letter: auction.letter, profile: state.profile, ...decision, elapsedMs,
    ownRack:[...ctx.myRack],balance:ctx.myBalance,score:ctx.myScore,remainingTiles:ctx.remainingTiles,
    observedRemainingMs: Number(auction.closesAt.microsSinceUnixEpoch / 1000n) - Date.now() });
  if (decision.amount <= 0) return;
  if (decision.amount > decision.baseline) {
    state.defensiveBids++;
    console.log(`[${name}] pressure ${decision.targetName}: ${auction.letter}, bid ${decision.amount} (base ${decision.baseline}, opportunity ${decision.threat}, ${decision.confidence})`);
  }
  connection.reducers.submitBid({ auctionId, amount: BigInt(decision.amount) }).catch((error) => fail("bid", error));
}

function play(state: State) {
  if (stopping || !state.ready || state.ending || state.pendingWord || Date.now() < state.retryWordAt || !connection) return;
  if (connection.db.matchState.id.find(state.id)?.status.tag !== "Running") return;
  if (state.lastSuccessfulRevision === state.rackRevision) return;
  const previous=state.victor?.snapshot.mode;
  const ctx=contextFor(state.id);
  const word = state.victor?state.victor.word({...ctx,opponents:opponentsFor(state.id)})
    :state.hercule?state.hercule.word({...ctx,publicKnownRack:new Map(),opponents:opponentsFor(state.id)}):(isWags ? chooseWagsWord : chooseHerculeWord)(ctx);
  logVictorMode(state,previous);
  if (!word) return;
  const revision = state.rackRevision;
  state.pendingWord = true;
  connection.reducers.submitWord({ matchId: state.id, word }).then(() => {
    state.lastSuccessfulRevision = revision;
    console.log(`[${name}] played ${word}`);
  }).catch((error) => { state.retryWordAt = Date.now() + 1000; fail(`word ${word}`, error); })
    .finally(() => { state.pendingWord = false; setImmediate(()=>play(state)); });
}

function finish(state: State) {
  if (finished.has(String(state.id)) || !connection) return;
  finished.add(String(state.id));
  const own = [...connection.db.matchParticipant.iter()].find((p) => p.matchId === state.id && p.botId === botId);
  if (!own) return stop("Missing final participant row", 1);
  const opponents = opponentsFor(state.id).map((p) => ({ id: p.id, name: p.name, score: p.score, simulated: p.isSimulated }));
  const result: MatchResult = {
    matchId: String(state.id), profile: state.profile, phase: state.phase, score: Number(own.score),
    rank: 1 + opponents.filter((p) => p.score > Number(own.score)).length, opponents,
    defensiveBids: state.defensiveBids, defensiveWins: state.defensiveWins, defensiveSpend: state.defensiveSpend, totalSpend: state.totalSpend,
    words: [...connection.db.wordPlay.iter()].filter((w) => w.matchId === state.id && w.botId === botId)
      .map((w) => ({ word: w.word, reward: Number(w.totalReward) })),
    ...(state.victor?{hybrid:{mode:state.victor.snapshot.mode,transitions:state.victor.snapshot.transitions}}:{}),
  };
  const done = study.complete(result);
  console.log(`[${name}] MATCH ${state.id} ${state.profile}: ${result.score} points, rank ${result.rank}/${opponents.length + 1}, ${opponents.filter((p) => !p.simulated).length} real rivals, ${state.defensiveWins} defensive wins`);
  state.subscription?.unsubscribe();
  states.delete(String(state.id));
  if (done || stopJoining) return stop(done ? evaluation?"Online test batch complete":"Practice batch complete" : "Practice time limit reached");
  setTimeout(joinLobby, 250);
}

function attachMatches() {
  if (!connection || botId === undefined || !subscribed || stopping) return;
  for (const participant of connection.db.matchParticipant.iter()) {
    if (participant.botId !== botId) continue;
    const id = participant.matchId;
    const key = String(id);
    const match = connection.db.matchState.id.find(id);
    if (match?.status.tag !== "Running" || states.has(key) || finished.has(key)) continue;
    const assignment = study.assign(key, match.currentRound > 1, normal);
    const state: State = { id, ...assignment, ready: false, ending: false, pendingWord: false, retryWordAt: 0,
      decisions: new Map(), handledResults: new Set(), defensiveBids: 0, defensiveWins: 0, defensiveSpend: 0, totalSpend: 0,
      rackRevision:0,lastSuccessfulRevision:-1, ...(isVictor?{victor:new VictorMateoPolicy()}:isWags?{}:{hercule:createLiveHerculePolicy()}) };
    states.set(key, state);
    state.subscription = connection.subscriptionBuilder().onError((ctx) => stop(`Match subscription failed: ${ctx.event}`, 1))
      .onApplied(() => {
        state.ready = true;
        console.log(`[${name}] match ${id}: ${state.profile} (${state.phase})`);
        if (connection!.db.matchState.id.find(id)?.status.tag === "Ended") {
          state.ending = true; setTimeout(() => finish(state), 150); return;
        }
        for (const auction of connection!.db.auction.iter()) if (auction.matchId === id && auction.status.tag === "Open") bid(auction.id);
        play(state);
      }).subscribe([
        `SELECT * FROM match_state WHERE id = ${id}`,
        ...["match_participant", "auction", "auction_result", "word_play", "my_rack"].map((table) => `SELECT * FROM ${table} WHERE match_id = ${id}`),
      ]);
  }
}

function joinLobby() {
  if (!connection || !subscribed || stopping || stopJoining || joinPending || states.size) return;
  const open = [...connection.db.lobby.iter()].find((lobby) => lobby.status.tag === "Open");
  if (!open) return;
  if ([...connection.db.lobbyMember.iter()].some((member) => member.botId === botId && member.lobbyId === open.id)) return;
  joinPending = true;
  connection.reducers.joinLobby({}).then(() => console.log(`[${name}] joined online lobby ${open.id}`))
    .catch((error) => fail("join lobby", error)).finally(() => { joinPending = false; });
}

function memberships() {
  const credential = connection!.db.botCredential.identity.find(connection!.identity!);
  if (!credential) return stop("Saved token is not a bot credential", 1);
  if(!isWags&&!isVictor){
    try{assertBotIdentity(name,connection!.db.bot.id.find(credential.botId)?.name);}
    catch(error){fail('Account identity',error);return stop('Online test account mismatch',1);}
  }
  if (isWags) {
    try { assertWagsIdentity(connection!.db.bot.id.find(credential.botId)?.name); }
    catch (error) { fail("Wags identity",error); return stop("Wags needs its own registered bot token",1); }
  }
  if(isVictor) {
    try {assertVictorIdentity(connection!.db.bot.id.find(credential.botId)?.name);}
    catch(error) {fail("VictorMateo identity",error);return stop("VictorMateo needs its own registered bot token",1);}
  }
  if (connectedToken) writeFileSync(tokenPath,connectedToken);
  botId = credential.botId;
  clearTimeout(connectionSetupTimeout);
  study.log('connected',{botId,name,profile:normal,adapterVersion:ONLINE_ADAPTER_VERSION});
  connection!.subscriptionBuilder().onError((ctx) => stop(`Lobby subscription failed: ${ctx.event}`, 1))
    .onApplied(() => {
      subscribed = true;
      attachMatches();
      if (process.env.TOURNAMENT_ID) connection!.reducers.registerForTournament({ tournamentId: BigInt(process.env.TOURNAMENT_ID) })
        .catch((error) => fail("tournament registration", error));
      joinLobby();
    }).subscribe([
      `SELECT match_state.* FROM match_state JOIN match_participant ON match_state.id = match_participant.match_id WHERE match_participant.bot_id = ${botId}`,
      `SELECT * FROM match_participant WHERE bot_id = ${botId}`,
      "SELECT * FROM lobby",
      `SELECT * FROM lobby_member WHERE bot_id = ${botId}`,
    ]);
}

const connectionSetupTimeout=setTimeout(()=>{
  if(botId===undefined)stop('Connection setup timed out; no confirmed account connection',1);
},45_000).unref();
connection = DbConnection.builder().withUri(host).withDatabaseName(database).withToken(token).withCompression('none')
  .onConnect((conn, _identity, freshToken) => {
    connectedToken = freshToken;
    conn.subscriptionBuilder().onError((ctx) => stop(`Account subscription failed: ${ctx.event}`, 1)).onApplied(() => {
      if (process.env.BOT_NONCE && !conn.db.botCredential.identity.find(conn.identity!)) {
        conn.reducers.claimCredential({ code: process.env.BOT_NONCE }).then(() => setTimeout(memberships, 150))
          .catch((error) => { fail("claim credential", error); stop("Credential claim failed", 1); });
      } else memberships();
    }).subscribe(["SELECT * FROM bot_credential", "SELECT * FROM bot", "SELECT * FROM bot_stats"]);
  })
  .onConnectError((_ctx, error) => stop(`Connection failed: ${error.message}`, 1))
  .onDisconnect(() => { if (!stopping) stop("Connection closed; saved completed results", 1); }).build();

connection.db.matchParticipant.onInsert(() => setImmediate(attachMatches));
connection.db.matchState.onInsert(() => setImmediate(attachMatches));
connection.db.matchState.onUpdate((_ctx, _old, match) => {
  const state = states.get(String(match.id));
  if (state && match.status.tag === "Ended" && !state.ending) {
    state.ending = true; setTimeout(() => finish(state), 150);
  } else setImmediate(attachMatches);
});
connection.db.auction.onInsert((_ctx, auction) => { if (auction.status.tag === "Open") setImmediate(() => bid(auction.id)); });
connection.db.auctionResult.onInsert((_ctx, result) => {
  const state = states.get(String(result.matchId));
  if (!state || state.handledResults.has(String(result.auctionId))) return;
  state.handledResults.add(String(result.auctionId));
  const decision = state.decisions.get(String(result.auctionId));
  if (result.winnerBotId === botId) {
    state.totalSpend += Number(result.paid);
    if (decision && decision.amount > decision.baseline) { state.defensiveWins++; state.defensiveSpend += Number(result.paid); }
  }
  study.log("auction-result", { matchId: result.matchId, auctionId: result.auctionId, winner: result.winnerBotId, paid: result.paid, letter: result.letter });
});
connection.db.wordPlay.onInsert((_ctx, word) => study.log("word", { matchId: word.matchId, botId: word.botId, word: word.word, reward: word.totalReward }));
{
  const rackChanged = (matchId: bigint) => {
    const state = states.get(String(matchId));
    if (state) { state.rackRevision++; setImmediate(()=>play(state)); }
  };
  connection.db.myRack.onInsert((_ctx,row)=>rackChanged(row.matchId));
  connection.db.myRack.onUpdate((_ctx,_old,row)=>rackChanged(row.matchId));
  connection.db.myRack.onDelete((_ctx,row)=>rackChanged(row.matchId));
}
setInterval(() => { for (const state of states.values()) play(state); }, isVictor?VICTOR_SETTINGS.pollMs:isWags ? WAGS_SETTINGS.pollMs : 300).unref();
setInterval(() => { attachMatches(); joinLobby(); }, 5000).unref();
if (training||evaluation) {
  setTimeout(() => {
    stopJoining = true;
    if (!states.size) stop("Practice time limit reached");
    else setTimeout(() => stop("Time limit reached during an unfinished match; completed results saved"), 120_000).unref();
  }, (evaluation?.minutes??25) * 60_000).unref();
}
process.on("SIGINT", () => stop("Stopped by user"));
process.on("SIGTERM", () => stop("Stopped by user"));
