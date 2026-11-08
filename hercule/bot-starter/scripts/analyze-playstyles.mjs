// Historical analysis only: no agent execution, simulations, or online reducers.
import { readFileSync,writeFileSync } from "node:fs";
import { join,resolve } from "node:path";
import { createHash } from "node:crypto";
const {directory}=JSON.parse(readFileSync("opponents/playstyle-snapshot.json","utf8"));
const manifest=JSON.parse(readFileSync(join(directory,"manifest.json"),"utf8"));
const table=(name)=>{const p=JSON.parse(readFileSync(join(directory,name+".json"),"utf8"))[0];const k=p.schema.elements.map(e=>e.name.some);return p.rows.map(row=>Object.fromEntries(k.map((key,i)=>[key,row[i]])));};
const group=(rows,key)=>{const map=new Map();for(const row of rows){if(!map.has(row[key]))map.set(row[key],[]);map.get(row[key]).push(row);}return map;};
const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
const quantile=(a,q)=>{if(!a.length)return null;const s=[...a].sort((x,y)=>x-y);return s[Math.floor((s.length-1)*q)];};
const time=value=>Number(value[0]);
const optional=value=>value[0]===0?value[1]:undefined;
const bots=new Map(table("bot").map(bot=>[bot.id,bot]));
const stats=table("bot_stats");
const versions=group(table("bot_version"),"bot_id");
const auctions=group(table("auction_result"),"match_id");
const words=group(table("word_play"),"match_id");
const roster=group(table("match_participant"),"match_id");
const tops=new Map(table("auction_top_bid_archive").map(row=>[row.auction_id,row.top_bid]));
const histories=new Map();
let audited=0,excluded=0;
const matches=table("match_state").filter(match=>match.status[0]===2).sort((a,b)=>a.id-b.id);

for(const match of matches){
  const rows=roster.get(match.id)??[];
  const rounds=[...(auctions.get(match.id)??[])].sort((a,b)=>time(a.closed_at)-time(b.closed_at)||a.auction_id-b.auction_id);
  const plays=[...(words.get(match.id)??[])].sort((a,b)=>time(a.played_at)-time(b.played_at)||a.id-b.id);
  if(!rows.length||!rounds.length)continue;
  const maxScore=Math.max(...rows.map(p=>p.score));
  const records=new Map(rows.map(p=>[p.bot_id,{botId:p.bot_id,matchId:match.id,createdAt:time(match.created_at),
    score:p.score,win:p.score===maxScore,finalBalance:p.balance,startingRack:match.starting_rack_size,
    realRivals:rows.filter(o=>o.bot_id!==p.bot_id&&bots.has(o.bot_id)&&!bots.get(o.bot_id).is_simulated).length,
    simulatedRivals:rows.filter(o=>o.bot_id!==p.bot_id&&bots.get(o.bot_id)?.is_simulated).length,
    unknownRivals:rows.filter(o=>o.bot_id!==p.bot_id&&!bots.has(o.bot_id)).length,
    spent:0,purchases:0,wordLengths:[],wordRewards:[],baseRewards:[],longRewards:0,shortRewards:0,
    wordProgress:[],longProgress:[],wordRackSizes:[],wordBalances:[],lateReward:0,lowCashWords:0,firstLongProgress:null,
    shortWords:0,longWords:0,wordLetters:0,firstLongPurchases:null,clearedRack:false,
    exactRackWindows:0,nonemptyRackWindows:0,observedWindows:0,lowCashWindows:0,
    winningBids:[],paid:[],winningFractions:[],nearSevenFractions:[],smallRackFractions:[],
    quickCashPurchases:0,priceByLetter:{},unaffordableAtOpening:0,firstPrice:match.auction_type[0]===0,
  }]));
  const state=new Map(rows.map(p=>[p.bot_id,{balance:100,score:0,size:match.starting_rack_size,known:new Map(),invalid:false}]));
  let nextWord=0;
  let completedAuctions=0;
  const apply=(cutoff,inclusive)=>{
    while(nextWord<plays.length&&(time(plays[nextWord].played_at)<cutoff||inclusive&&time(plays[nextWord].played_at)===cutoff)){
      const play=plays[nextWord++],s=state.get(play.bot_id),r=records.get(play.bot_id);
      if(!s)continue;
      const progress=completedAuctions/Math.max(1,rounds.length);
      r.wordProgress.push(progress);r.wordRackSizes.push(s.size);r.wordBalances.push(s.balance);
      if(progress>=.8)r.lateReward+=play.total_reward;
      if(s.balance<=20)r.lowCashWords++;
      r.wordLengths.push(play.word.length);r.wordRewards.push(play.total_reward);r.baseRewards.push(play.base_score);
      r.wordLetters+=play.word.length;
      if(play.word.length>=7){r.longWords++;r.longRewards+=play.total_reward;r.longProgress.push(progress);if(r.firstLongPurchases===null){r.firstLongPurchases=r.purchases;r.firstLongProgress=progress;}}
      if(play.word.length<=4){r.shortWords++;r.shortRewards+=play.total_reward;}
      s.balance+=play.total_reward;s.score+=play.total_reward;s.size-=play.word.length;
      if(s.size<0)s.invalid=true;
      if(s.size===0)r.clearedRack=true;
      for(const letter of play.word)s.known.set(letter,Math.max(0,(s.known.get(letter)??0)-1));
    }
  };
  let opens=optional(match.started_at)?.[0]??time(match.created_at);
  for(const auction of rounds){
    apply(opens,true);
    for(const [id,s]of state){const r=records.get(id);r.observedWindows++;if(s.balance<=10)r.lowCashWindows++;
      if(s.size>0){r.nonemptyRackWindows++;if([...s.known.values()].reduce((a,b)=>a+b,0)===s.size)r.exactRackWindows++;}}
    const winnerId=optional(auction.winner_bot_id),winner=state.get(winnerId),record=records.get(winnerId);
    if(record){
      const bid=tops.get(auction.auction_id)??(match.auction_type[0]===0?auction.paid:undefined);
      if(bid!==undefined){record.winningBids.push(bid);if(winner.balance>0&&bid<=winner.balance){const fraction=bid/winner.balance;record.winningFractions.push(fraction);
        if(winner.size>=6)record.nearSevenFractions.push(fraction);if(winner.size<=4)record.smallRackFractions.push(fraction);
      }else if(bid>winner.balance)record.unaffordableAtOpening++;}
      record.paid.push(auction.paid);
      (record.priceByLetter[auction.letter]??=[]).push(auction.paid);
      const after=plays.find(play=>play.bot_id===winnerId&&time(play.played_at)>=time(auction.closed_at)&&
        time(play.played_at)-time(auction.closed_at)<=2*Number(match.auction_duration_ms)*1000&&play.word.includes(auction.letter));
      if(after)record.quickCashPurchases++;
    }
    const closes=time(auction.closed_at);apply(closes,false);
    if(winner){record.spent+=auction.paid;record.purchases++;winner.balance-=auction.paid;winner.size++;winner.known.set(auction.letter,(winner.known.get(auction.letter)??0)+1);if(winner.balance<0)winner.invalid=true;}
    completedAuctions++;
    apply(closes,true);opens=closes;
  }
  apply(Infinity,true);
  const valid=rows.every(p=>{const s=state.get(p.bot_id);return !s.invalid&&s.balance===p.balance&&s.score===p.score;});
  if(!valid){excluded++;continue;}audited++;
  for(const [id,record]of records){if(!histories.has(id))histories.set(id,[]);histories.get(id).push(record);}
}

function aggregate(games){
  const wordLengths=games.flatMap(g=>g.wordLengths),rewards=games.flatMap(g=>g.wordRewards),paid=games.flatMap(g=>g.paid),bids=games.flatMap(g=>g.winningBids);
  const histogram={};for(const length of wordLengths)histogram[length]=(histogram[length]??0)+1;
  const modal=Number(Object.entries(histogram).sort((a,b)=>b[1]-a[1])[0]?.[0]??0);
  const score=games.reduce((sum,g)=>sum+g.score,0),spend=games.reduce((sum,g)=>sum+g.spent,0);
  const knownWindows=games.reduce((sum,g)=>sum+g.nonemptyRackWindows,0),allWindows=games.reduce((sum,g)=>sum+g.observedWindows,0);
  return {games:games.length,wins:games.filter(g=>g.win).length,winRate:mean(games.map(g=>Number(g.win))),score:mean(games.map(g=>g.score)),
    spend:mean(games.map(g=>g.spent)),purchases:mean(games.map(g=>g.purchases)),wordsPerGame:wordLengths.length/Math.max(1,games.length),
    meanWordLength:mean(wordLengths),modalWordLength:modal,longWordFraction:games.reduce((s,g)=>s+g.longWords,0)/Math.max(1,wordLengths.length),
    shortWordFraction:games.reduce((s,g)=>s+g.shortWords,0)/Math.max(1,wordLengths.length),longScoreFraction:games.reduce((s,g)=>s+g.longRewards,0)/Math.max(1,score),
    shortScoreFraction:games.reduce((s,g)=>s+g.shortRewards,0)/Math.max(1,score),bonusFraction:1-games.flatMap(g=>g.baseRewards).reduce((a,b)=>a+b,0)/Math.max(1,score),
    medianPaid:quantile(paid,.5),paidP90:quantile(paid,.9),medianWinningBid:quantile(bids,.5),bidP90:quantile(bids,.9),
    medianWinningBalanceFraction:quantile(games.flatMap(g=>g.winningFractions),.5),
    nearSevenWinningBalanceFraction:quantile(games.flatMap(g=>g.nearSevenFractions),.5),
    smallRackWinningBalanceFraction:quantile(games.flatMap(g=>g.smallRackFractions),.5),
    medianFinalBalance:quantile(games.map(g=>g.finalBalance),.5),payoutPerCoinSpent:spend?score/spend:null,
    lowCashFraction:games.reduce((s,g)=>s+g.lowCashWindows,0)/Math.max(1,allWindows),
    emptiedRackGameFraction:mean(games.map(g=>Number(g.clearedRack))),
    exactNonemptyRackFraction:games.reduce((s,g)=>s+g.exactRackWindows,0)/Math.max(1,knownWindows),
    medianPurchasesToFirstLong:quantile(games.map(g=>g.firstLongPurchases).filter(v=>v!==null),.5),
    medianFirstLongProgress:quantile(games.map(g=>g.firstLongProgress).filter(v=>v!==null),.5),
    medianWordProgress:quantile(games.flatMap(g=>g.wordProgress),.5),medianWordRackSize:quantile(games.flatMap(g=>g.wordRackSizes),.5),
    medianBalanceAtWord:quantile(games.flatMap(g=>g.wordBalances),.5),lowCashWordFraction:games.reduce((s,g)=>s+g.lowCashWords,0)/Math.max(1,wordLengths.length),
    lateScoreFraction:games.reduce((s,g)=>s+g.lateReward,0)/Math.max(1,score),
    quickCashPurchaseFraction:games.reduce((s,g)=>s+g.quickCashPurchases,0)/Math.max(1,games.reduce((s,g)=>s+g.purchases,0)),
    firstPriceGames:games.filter(g=>g.firstPrice).length,histogram,
    winningGameLongScoreFraction:games.filter(g=>g.win).reduce((s,g)=>s+g.longRewards,0)/Math.max(1,games.filter(g=>g.win).reduce((s,g)=>s+g.score,0)),
    losingGameLongScoreFraction:games.filter(g=>!g.win).reduce((s,g)=>s+g.longRewards,0)/Math.max(1,games.filter(g=>!g.win).reduce((s,g)=>s+g.score,0))};
}
const summaries=[];
for(const stat of stats){
  const bot=bots.get(stat.bot_id);if(!bot||bot.is_simulated||stat.matches_played<10)continue;
  const history=versions.get(bot.id)??[];
  const cutoff=Math.max(0,...history.map(v=>time(v.started_at)));
  const recent=(histories.get(bot.id)??[]).filter(g=>g.createdAt>=cutoff).sort((a,b)=>b.matchId-a.matchId).slice(0,100);
  const competitive=recent.filter(g=>g.realRivals>=4);
  summaries.push({botId:bot.id,name:bot.name,version:Math.max(1,...history.map(v=>v.version)),globalMatches:stat.matches_played,globalWins:stat.wins,
    globalWinRate:stat.wins/stat.matches_played,rating:stat.rating,openskillMu:stat.openskill_mu,
    recentMatchIds:recent.map(g=>g.matchId),recent:aggregate(recent),competitive:aggregate(competitive)});
}
summaries.sort((a,b)=>b.globalWinRate-a.globalWinRate);
const substantial=summaries.filter(bot=>bot.globalMatches>=50);
const chosen=new Set([...substantial.slice(0,12).map(bot=>bot.botId),...[...substantial].sort((a,b)=>b.rating-a.rating).slice(0,6).map(bot=>bot.botId),69]);
const selected=summaries.filter(bot=>chosen.has(bot.botId));
const out={collectedAt:manifest.sources.map(s=>s.collectedAt).sort().at(-1),snapshot:directory,
  audit:{completedMatches:matches.length,valid:audited,excluded},selection:"Top 12 current win rates among bots with at least 50 games, plus top six ratings; latest 100 audited current-version games per bot. Competitive subset requires at least four known real rivals.",
  summaries,selected,sources:manifest.sources.map(s=>({...s,sha256:createHash("sha256").update(readFileSync(join(directory,s.table+".json"))).digest("hex")}))};
writeFileSync(join(directory,"playstyle-analysis.json"),JSON.stringify(out,null,2));
writeFileSync("opponents/playstyle-analysis.json",JSON.stringify(out,null,2));
const pct=x=>x===null?"n/a":(100*x).toFixed(1)+"%",f=x=>x===null?"n/a":x.toFixed(1),esc=s=>s.replaceAll("|","\\|");
const lines=["# Strategy evidence from public history","",`Snapshot: ${out.collectedAt}. ${out.selection}`,"",
  `Audited ${audited} completed games; ${excluded} failed score/balance consistency checks and were excluded. No matches were run and no strategy settings were changed.`,"",
  "| Bot | Current public wins/games | Recent games | Recent score | Competitive wins/games | 7+ letter score share | Modal length | Purchases/game | Coins spent/game | Median final coins |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ...selected.map(b=>`| ${esc(b.name)} | ${b.globalWins}/${b.globalMatches} (${pct(b.globalWinRate)}) | ${b.recent.games} | ${f(b.recent.score)} | ${b.competitive.wins}/${b.competitive.games} | ${pct(b.recent.longScoreFraction)} | ${b.recent.modalWordLength} | ${f(b.recent.purchases)} | ${f(b.recent.spend)} | ${f(b.recent.medianFinalBalance)} |`),"",
  "## Liquidity, timing, and public rack visibility","",
  "| Bot | Median winning bid / paid | Winning bid as balance % | Cash ≤10 during auctions | Nonempty rack exactly reconstructible | Games with rack emptied | Median buys until first 7+ word | Win followed by word within two auction periods using that letter |",
  "|---|---:|---:|---:|---:|---:|---:|---:|",
  ...selected.map(b=>`| ${esc(b.name)} | ${f(b.recent.medianWinningBid)} / ${f(b.recent.medianPaid)} | ${pct(b.recent.medianWinningBalanceFraction)} | ${pct(b.recent.lowCashFraction)} | ${pct(b.recent.exactNonemptyRackFraction)} | ${pct(b.recent.emptiedRackGameFraction)} | ${f(b.recent.medianPurchasesToFirstLong)} | ${pct(b.recent.quickCashPurchaseFraction)} |`),"",
  "Exact rack visibility is established only when public rack size equals the total conservative known letter counts. Initial letters are never guessed. Emptying a rack guarantees a fresh exact public reconstruction thereafter. No private tables or live winning-bid spectator views were queried.","",
  "## Observed payout timing","",
  "| Bot | Median first 7+ word, auction progress | Median rack size before word | Median coins before word | Words played at ≤20 coins | Points scored in final 20% of auctions |",
  "|---|---:|---:|---:|---:|---:|",
  ...selected.map(b=>`| ${esc(b.name)} | ${pct(b.recent.medianFirstLongProgress)} | ${f(b.recent.medianWordRackSize)} | ${f(b.recent.medianBalanceAtWord)} | ${pct(b.recent.lowCashWordFraction)} | ${pct(b.recent.lateScoreFraction)} |`),"",
  "Progress counts settled auctions, rather than elapsed wall time. First-long-word medians include only games in which that bot played a 7+ word. The records show when words were played, not when they first became available, so these timings do not establish deliberate waiting or a particular cash-out rule. Low-cash auction statistics sample balances at auction opening; they do not measure every moment within the auction.","",
  "The purchase-to-word statistic is an association: it does not prove the purchased letter was needed, because an identical letter might already have been held privately. Winning-bid ratios describe won auctions only; losing bids remain censored.","",
  "Public leaderboard percentages include ties, use each bot's current version, and span different eras and opponent fields. The recent subset can differ substantially; competitive subsets are not strength-adjusted and do not isolate causality. Deleted opponent identities are marked unknown rather than assumed real. Short-lived new entrants are excluded from the substantial-history selection.","",
  `Raw and aggregate evidence: ${join(directory,"playstyle-analysis.json")}.`,"",
];
writeFileSync(resolve("../PLAYSTYLE-EVIDENCE.md"),lines.join("\n"));
console.log(JSON.stringify({audit:out.audit,selected:selected.map(b=>({name:b.name,winRate:b.globalWinRate,rating:b.rating,
  games:b.recent.games,score:b.recent.score,competitive:`${b.competitive.wins}/${b.competitive.games}`,longShare:b.recent.longScoreFraction,
  meanLength:b.recent.meanWordLength,spent:b.recent.spend,paid:b.recent.medianPaid,bid:b.recent.medianWinningBid,
  final:b.recent.medianFinalBalance,lowCash:b.recent.lowCashFraction,exact:b.recent.exactNonemptyRackFraction,cleared:b.recent.emptiedRackGameFraction,
  near7:b.recent.nearSevenWinningBalanceFraction,smallRack:b.recent.smallRackWinningBalanceFraction,quickCash:b.recent.quickCashPurchaseFraction}))},null,2));
