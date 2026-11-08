import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, existsSync, unlinkSync, rmdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { Study, metric, selectProfile, type MatchResult, type Profile } from "./study.js";

const game = (profile: Profile, score: number, rival: number): MatchResult => ({
  matchId: "1", profile, phase: "exploration", score, rank: 2,
  opponents: [{ id: "2", name: "rival", score: rival, simulated: false }],
  defensiveBids: 0, defensiveWins: 0, defensiveSpend: 0, totalSpend: 0, words: [],
});
test("tuning favors suppressing the strongest rival and keeps validation separate", () => {
  const cautious = game("cautious", 100, 100);
  const assertive = game("assertive", 80, 60);
  assert.equal(selectProfile([cautious, cautious, assertive, assertive]), "assertive");
  assert.equal(selectProfile([cautious, cautious, { ...assertive, phase: "validation" }]), "cautious");
  const pressure = game("pressure", 90, 30);
  assert.equal(selectProfile([cautious, cautious, assertive, assertive, pressure, pressure]), "pressure");
});
test("simulated-only and partial games are excluded from the real-opponent objective", () => {
  const result = game("assertive", 1000, 0);
  assert.equal(metric({ ...result, phase: "partial" }), null);
  assert.equal(metric({ ...result, opponents: result.opponents.map((opponent) => ({ ...opponent, simulated: true })) }), null);
  assert.equal(selectProfile([result]), "cautious");
});

test("resumes a saved study without duplicating completed games or changing its plan", (t) => {
  const priorPlan = process.env.BOT_TRAIN_PLAN;
  const priorResume = process.env.BOT_TRAIN_RESUME;
  delete process.env.BOT_TRAIN_RESUME;
  process.env.BOT_TRAIN_PLAN = "pressure,pressure";
  const root = mkdtempSync(join(tmpdir(), "scrabble-study-"));
  const initial = new Study(true, 4, root);
  t.after(() => {
    if (priorPlan === undefined) delete process.env.BOT_TRAIN_PLAN; else process.env.BOT_TRAIN_PLAN = priorPlan;
    if (priorResume === undefined) delete process.env.BOT_TRAIN_RESUME; else process.env.BOT_TRAIN_RESUME = priorResume;
    for (const file of ["events.jsonl", "results.json", "REPORT.md"]) {
      const path = join(initial.dir, file);
      if (existsSync(path)) unlinkSync(path);
    }
    rmdirSync(initial.dir);
    rmdirSync(root);
  });
  initial.assign("1", false);
  initial.complete({ ...game("pressure", 25, 100), matchId: "1" });
  initial.assign("2", false);
  initial.complete({ ...game("pressure", 30, 110), matchId: "2" });
  delete process.env.BOT_TRAIN_PLAN;
  process.env.BOT_TRAIN_RESUME = basename(initial.dir);
  const resumed = new Study(true, 4, root);
  assert.equal(resumed.results.length, 2);
  assert.deepEqual(resumed.assign("3", false), { profile: "pressure", phase: "validation" });
  resumed.complete({ ...game("pressure", 30, 110), matchId: "2" });
  assert.equal(resumed.results.length, 2);
  const savedPath=join(initial.dir,"results.json");
  const oldVersion=JSON.parse(readFileSync(savedPath,"utf8"));
  delete oldVersion.policyVersion;
  writeFileSync(savedPath,JSON.stringify(oldVersion));
  assert.throws(()=>new Study(true,4,root),/different policy version/);
});
