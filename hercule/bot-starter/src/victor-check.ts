// Scripted decisions only. No connection, credentials, or match simulation.
import {loadDictionary} from "./dictionary.js";
import {prepareDictionary} from "./strategy.js";
import {VictorMateoPolicy} from "./victor.js";

const dictionary=loadDictionary();prepareDictionary(dictionary);
const policy=new VictorMateoPolicy();
const rack=(text:string)=>{const result=new Map<string,number>();for(const letter of text)result.set(letter,(result.get(letter)??0)+1);return result;};
const opponent={id:"rival",name:"Nijello",score:30,balance:60,isSimulated:false,knownRack:rack("QUI"),rackSize:3};
const context={myRack:rack("AT"),dictionary,myBalance:100,myScore:0,opponents:[opponent],remainingTiles:30,progress:0.3};
console.log("VictorMateo offline check: scripted decisions, no connection or games.");
for(const snapshot of [
  {...context,letter:"E"},
  {...context,myBalance:130,myScore:60,letter:"E"},
  {...context,myBalance:130,myScore:60,letter:"Z"},
  {...context,myBalance:40,myScore:60,letter:"E"},
]) {
  const result=policy.bid(snapshot);
  console.log(`${result.mode}: ${snapshot.myBalance} coins, ${snapshot.myScore} points; bid ${result.amount} for ${snapshot.letter}. ${result.modeReason}`);
}
console.log("These snapshots verify switching behavior; they are not match-performance results.");
