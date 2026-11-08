import { loadDictionary } from "./dictionary.js";
import { prepareDictionary, wordReward } from "./strategy.js";
import { createLiveHerculePolicy,HERCULE_LIVE_VERSION } from "./hercule-live.js";

const dictionary = loadDictionary();
prepareDictionary(dictionary);
const myRack = new Map([["Q", 1], ["U", 1], ["I", 1]]);
const policy=createLiveHerculePolicy();
const start = performance.now();
const bid = policy.bid({ letter: "Z", myBalance: 100, myScore:0,myRack,publicKnownRack:new Map(),opponents:[],dictionary,remainingTiles:30,progress:.3,auctionType:'Vickrey' });
myRack.set("Z", 1);
const word = policy.word({ myRack,publicKnownRack:new Map(),opponents:[],myBalance:100,myScore:0,dictionary,remainingTiles:29,progress:.3,auctionType:'Vickrey' });
console.log(`Loaded ${dictionary.length.toLocaleString()} dictionary words.`);
console.log(`hercule-v1: ${HERCULE_LIVE_VERSION}; defense disabled; median buying and existing word timing.`);
console.log(`Example: holding Q U I, bid ${bid} for Z.`);
console.log(`With Z: play ${word}, worth ${word ? wordReward(word) : 0} points.`);
console.log(`Two decisions took ${(performance.now() - start).toFixed(1)} ms. No server connection made.`);
