import assert from "node:assert/strict";
import { test } from "node:test";
import { defensiveBid, knownRacks, type Opponent } from "./defense.js";

const rack = (text: string) => {
  const result = new Map<string, number>();
  for (const letter of text) result.set(letter, (result.get(letter) ?? 0) + 1);
  return result;
};
const ctx = { letter: "Z", myBalance: 100, myRack: rack("AT"), dictionary: ["AT", "QUIZ"] };
const rival: Opponent = { id: "2", name: "rival", balance: 100, score: 50, isSimulated: false, knownRack: rack("QUI") };

test("blocks a rival's observable valuable completion and can reproduce baseline", () => {
  const defense = defensiveBid(ctx, [rival], 1.6);
  assert.ok(defense.amount > defense.baseline);
  assert.equal(defense.target, rival.id);
  assert.equal(defense.threat, 33);
  assert.equal(defensiveBid(ctx, [rival], 0).amount, defense.baseline);
});

test("does not invent starting letters, target simulations, or spend the cash reserve", () => {
  assert.equal(defensiveBid(ctx, [{ ...rival, knownRack: rack("Q") }], 0.8).threat, 0);
  const simulated = defensiveBid(ctx, [{ ...rival, isSimulated: true }], 0.8);
  assert.equal(simulated.amount, simulated.baseline);
  for (const myBalance of [0, 1, 20, 21, 25, 100]) {
    const decision = defensiveBid({ ...ctx, myBalance }, [rival], 0.8);
    assert.ok(decision.amount <= myBalance);
    assert.ok(myBalance-decision.amount>=Math.min(12,Math.floor(myBalance*0.15)));
  }
});

test("reconstructs retained public purchases in time order and subtracts repeated letters", () => {
  const result = knownRacks([
    { time: 3n, kind: "word", botId: "2", text: "TEE" },
    { time: 1n, kind: "purchase", botId: "2", text: "T" },
    { time: 2n, kind: "purchase", botId: "2", text: "E" },
    { time: 4n, kind: "purchase", botId: "2", text: "E" },
  ]).get("2")!;
  assert.equal(result.get("T"), 0);
  assert.equal(result.get("E"), 1);
});
