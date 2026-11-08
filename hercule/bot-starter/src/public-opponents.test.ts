import assert from "node:assert/strict";
import { test } from "node:test";
import { chronologicalSplit, wilsonInterval, bidFeatures, BID_FEATURES, fitBid, predictBid, evaluateBid, createSampleBot,
  type BidSample, type PublicOpponentModel } from "./public-opponents.js";
import { chooseWord, chooseWordAtLeast } from "./strategy.js";

const state = { letter: "A", balance: 100, score: 10, rackSize: 5, knownCopies: 0, progress: 0.2, leaderScore: 20 };

test("holds out newer whole matches, with no match overlap", () => {
  const split = chronologicalSplit([5, 1, 3, 2, 4, 3]);
  assert.deepEqual(split, { train: [1,2,3,4], test: [5] });
  assert.ok(split.train.every((id) => !split.test.includes(id)));
});
test("win-rate intervals distinguish ten games from a substantial history", () => {
  const small = wilsonInterval(6,10), large = wilsonInterval(60,100);
  assert.ok(small[1]-small[0] > large[1]-large[0]);
  assert.deepEqual(wilsonInterval(0,0), [0,1]);
});
test("fits exact wins while treating losses as upper bounds rather than zero bids", () => {
  const samples: BidSample[] = Array.from({ length: 80 }, (_, i) => ({ ...state, matchId: i,
    kind: i % 2 ? "upper" : "exact", amount: i % 2 ? 30 : 20 }));
  const model = fitBid(samples);
  assert.ok(predictBid(model,state) >= 17 && predictBid(model,state) <= 23);
  assert.equal(evaluateBid(model,samples).upperViolationRate, 0);
});
test("bids remain integer and affordable with poor or extreme fitted values", () => {
  const model = fitBid([{ ...state, matchId: 1, kind: "exact", amount: 20 }],"letter-median");
  assert.equal(bidFeatures(state).length,BID_FEATURES.length);
  for (const balance of [0,1,5,100,1000]) {
    const bid = predictBid(model,{ ...state,balance });
    assert.ok(Number.isInteger(bid) && bid >= 0 && bid <= balance);
  }
});
test("word length constraints preserve repeated tiles and leave default scoring unchanged", () => {
  const ctx = { myRack: new Map([["C",1],["A",1],["T",1],["X",1]]), dictionary: ["CAT","AX","TACT"] };
  assert.equal(chooseWord(ctx),"AX");
  assert.equal(chooseWordAtLeast(ctx,3),"CAT");
  assert.equal(chooseWordAtLeast(ctx,4),null);
});
test("sample-bot word policy uses its own rack, with explicit cash-out fallback", () => {
  const model = { name: "teacher", bid: fitBid([],"letter-median"),
    word: { minimumLength: 3, cashOutBalance: 20, flushRemaining: 3, rackLimit: 14 } } as PublicOpponentModel;
  const bot = createSampleBot(model);
  const ctx = { myRack: new Map([["A",1],["X",1]]), myBalance: 100, myScore: 0,
    dictionary: ["AX","CAT"], opponents: [], progress: 0.3, remainingTiles: 30 };
  assert.equal(bot.chooseWord(ctx),null);
  assert.equal(bot.chooseWord({ ...ctx,myBalance: 10 }),"AX");
});
