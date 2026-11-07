// Checks for the default strategy. Run with `npm test`.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import { chooseWord, decideBid, decideBidAsync } from "./strategy.js";
import { benoitModel } from "./model_inference.js";

const rack = (letters: string) => {
  const r = new Map<string, number>();
  for (const c of letters) r.set(c, (r.get(c) ?? 0) + 1);
  return r;
};

test("decideBid produces competitive bids and respects balance", async () => {
  const r = rack("RETI");
  const bidSync = decideBid({ letter: "E", myBalance: 100, myRack: r });
  assert.ok(bidSync >= 10, `Expected competitive bid >= 10, got ${bidSync}`);

  // Test async ONNX-backed bid
  const bidAsync = await decideBidAsync({ letter: "E", myBalance: 100, myRack: r });
  assert.ok(bidAsync >= 10, `Expected async bid >= 10, got ${bidAsync}`);
  assert.ok(bidAsync <= 100, `Expected bid <= balance, got ${bidAsync}`);
});

test("chooseWord picks the longest word the rack can spell", () => {
  const dictionary = ["TACT", "CAT", "AT"];
  assert.equal(chooseWord({ myRack: rack("TACT"), dictionary }), "TACT");
  assert.equal(chooseWord({ myRack: rack("CATX"), dictionary }), "CAT");
  assert.equal(chooseWord({ myRack: rack("CX"), dictionary }), null);
});

// The bot can't see a match move on while it searches, so a slow search
// means bids on closed auctions and words from letters it already spent.
test("chooseWord searches the whole wordlist well within an auction", () => {
  const dictionary = fs
    .readFileSync(
      new URL("../../spacetimedb/wordlist.txt", import.meta.url),
      "utf8",
    )
    .split("\n")
    .map((l) => l.trim().toUpperCase())
    .filter((l) => l.length >= 2)
    .sort((a, b) => b.length - a.length);
  // The largest rack an admin can set, with almost nothing to spell.
  const myRack = rack("BCDFGHJKLMNPQRSTVWXZ");
  const start = performance.now();
  chooseWord({ myRack, dictionary });
  assert.ok(performance.now() - start < 100, "a search took 100 ms or more");
});
