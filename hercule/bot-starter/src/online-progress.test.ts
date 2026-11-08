import assert from 'node:assert/strict';
import {test} from 'node:test';
import {auctionProgress} from './online-progress.js';
test('live progress agrees with the local arena throughout a full game',()=>{
  for(const auctions of [1,38,68,86])for(let round=1;round<=auctions;round++){
    const view=auctionProgress({currentRound:round,bagTotal:auctions-round});
    assert.equal(view.remainingTiles,auctions-round);
    assert.equal(view.progress,(round-1)/Math.max(1,auctions-1));
  }
});
test('a shrinking bag is not dealt or counted down a second time',()=>{
  assert.deepEqual(auctionProgress({currentRound:1,bagTotal:67}),{remainingTiles:67,progress:0});
  assert.deepEqual(auctionProgress({currentRound:19,bagTotal:49}),{remainingTiles:49,progress:18/67});
  assert.deepEqual(auctionProgress({currentRound:68,bagTotal:0}),{remainingTiles:0,progress:1});
});
