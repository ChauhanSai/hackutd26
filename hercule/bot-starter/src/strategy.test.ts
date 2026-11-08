import assert from "node:assert/strict";
import { test } from "node:test";
import { loadDictionary } from "./dictionary.js";
import { chooseWord, decideBid, prepareDictionary, wordReward } from "./strategy.js";

const rack = (letters: string) => {
  const r = new Map<string, number>();
  for (const c of letters) r.set(c, (r.get(c) ?? 0) + 1);
  return r;
};

test("plays available words and returns null when none are possible", () => {
  const dictionary = ["TACT", "CAT", "AT"];
  assert.equal(chooseWord({ myRack: rack("TACT"), dictionary }), "TACT");
  assert.equal(chooseWord({ myRack: rack("CATX"), dictionary }), "CAT");
  assert.equal(chooseWord({ myRack: rack("CX"), dictionary }), null);
});

test("chooses actual reward over length without assuming dictionary order", () => {
  assert.equal(chooseWord({ myRack: rack("CATX"), dictionary: ["CAT", "AX"] }), "AX");
});

test("requires repeated letters and notices a rack changed in place", () => {
  const dictionary = ["TEE"];
  const myRack = rack("TE");
  assert.equal(chooseWord({ myRack, dictionary }), null);
  myRack.set("E", 2);
  assert.equal(chooseWord({ myRack, dictionary }), "TEE");
});

test("rewards match the server multipliers and round down", () => {
  assert.equal(wordReward("AX"), 9);
  assert.equal(wordReward("QUIZ"), 33);
  assert.equal(wordReward("JAZZ"), 43);
  assert.equal(wordReward("CART"), 9);
  assert.equal(wordReward("RATES"), 10);
  assert.equal(wordReward("RETAIN"), 15);
  assert.equal(wordReward("RETAINS"), 21);
});

test("bids more for a completion than an unrelated expensive letter", () => {
  const ctx = { myRack: rack("QUI"), dictionary: ["QUIZ"], myBalance: 100 };
  assert.ok(decideBid({ ...ctx, letter: "Z" }) > decideBid({ ...ctx, letter: "Q" }));
});

test("bids stay integral, affordable, and preserve cash", () => {
  for (const myBalance of [0, 1, 2, 10, 100, 100.5, -1, NaN]) {
    const bid = decideBid({ myRack: rack("QUI"), dictionary: ["QUIZ"], myBalance, letter: "Z" });
    assert.ok(Number.isInteger(bid) && bid >= 0);
    if (Number.isFinite(myBalance) && myBalance >= 0) assert.ok(bid <= myBalance);
    if (myBalance >= 10) assert.ok(bid <= Math.floor(myBalance * 0.35));
  }
});

test("a repeated letter completes a word but two missing tiles do not", () => {
  const ctx = { myRack: rack("LETER"), dictionary: ["LETTER"], myBalance: 100 };
  assert.ok(decideBid({ ...ctx, letter: "T" }) > decideBid({ ...ctx, letter: "E" }));
  assert.equal(chooseWord({ myRack: rack("LEER"), dictionary: ["LETTER"] }), null);
});

test("full dictionary decisions fit comfortably inside a one-second auction", (t) => {
  const dictionary = loadDictionary();
  prepareDictionary(dictionary);
  const ctx = { myRack: rack("BCDFGHJKLMNPQRSTVWXZ"), dictionary };
  const start = performance.now();
  chooseWord(ctx);
  decideBid({ ...ctx, letter: "A", myBalance: 100 });
  const elapsed = performance.now() - start;
  t.diagnostic(`Full dictionary word + bid search: ${elapsed.toFixed(1)} ms`);
  assert.ok(elapsed < 100, `a decision took ${elapsed.toFixed(1)} ms`);
});
