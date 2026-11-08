// Offline behavior demonstration only: no server, credentials, or match loop.
import { loadDictionary } from "./dictionary.js";
import { wordReward } from "./strategy.js";
import { defensiveBid } from "./defense.js";
import { decideWagsBid, chooseWagsWord, prepareWagsDictionary } from "./wags.js";

const dictionary = loadDictionary();
prepareWagsDictionary(dictionary);
const myRack = new Map<string,number>();
for (const letter of "TACTICSAX") myRack.set(letter,(myRack.get(letter)??0)+1);
console.log("Wags offline check — no connection or games.");
for (const letter of ["A","E","X","Z"]) {
  const ctx = { letter,myBalance:100,myRack:new Map<string,number>(),dictionary };
  console.log(`${letter}: Wags bids ${decideWagsBid(ctx)}; hercule-v1's own-acquisition bid is ${defensiveBid(ctx,[],1.6).amount}.`);
}
const played: string[] = [];
let score = 0;
for (let i = 0; i < 10; i++) {
  const word = chooseWagsWord({ myRack,dictionary });
  if (!word) break;
  played.push(word);
  score += wordReward(word);
  for (const letter of word) myRack.set(letter,myRack.get(letter)!-1);
}
console.log(`From TACTICSAX: ${played.join(" → ")} (${score} points).`);
