import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { VICTOR_POLICY_VERSION, type VictorMode } from "./victor.js";
import {SCORING_POLICY_VERSION} from "./scoring-v3.js";
import {HERCULE_POLICY_VERSION} from "./hercule-policy.js";

export const PROFILES = { baseline: 0, cautious: 0.4, assertive: 0.8, pressure: 1.6, wags: 0, victor:0, hercule:0 } as const;
export type BotStrategy="defensive"|"wags"|"victor"|"hercule";
export type Profile = keyof typeof PROFILES;
export interface MatchResult {
  matchId: string;
  profile: Profile;
  phase: "exploration" | "validation" | "normal" | "partial";
  score: number;
  rank: number;
  opponents: { id: string; name: string; score: number; simulated: boolean }[];
  defensiveBids: number;
  defensiveWins: number;
  defensiveSpend: number;
  totalSpend: number;
  words: { word: string; reward: number }[];
  hybrid?: {mode:VictorMode;transitions:number};
}

export function metric(result: MatchResult): number | null {
  const real = result.opponents.filter((opponent) => !opponent.simulated);
  if (!real.length || result.phase === "partial") return null;
  // Penalize the strongest real rival more than we reward our own points.
  return 0.5 * result.score - Math.max(...real.map((opponent) => opponent.score));
}

export function selectProfile(results: MatchResult[]): Profile {
  const eligible = (["cautious", "assertive", "pressure"] as Profile[]).map((profile) => {
    const games = results.filter((result) => result.profile === profile && metric(result) !== null && result.phase === "exploration");
    return { profile, games, average: games.reduce((sum, result) => sum + metric(result)!, 0) / Math.max(1, games.length) };
  }).filter((item) => item.games.length >= 2);
  eligible.sort((a, b) => b.average - a.average);
  return eligible[0]?.profile ?? "cautious";
}

export class Study {
  readonly dir: string;
  readonly results: MatchResult[] = [];
  readonly training: boolean;
  readonly maxMatches: number;
  private assignments = new Map<string, { profile: Profile; phase: MatchResult["phase"] }>();
  private completed = new Set<string>();
  private next = 0;
  private schedule: Profile[];
  private botName: string;
  private strategy: BotStrategy;
  private fixedProfile?:Profile;
  private policyVersion:string;

  constructor(training: boolean, maxMatches: number, root = "training", context:{botName:string;strategy:BotStrategy;fixedProfile?:Profile;policyVersion?:string} = { botName: "hercule-v1", strategy: "defensive" }) {
    if(training&&context.strategy!=="defensive"&&context.strategy!=="hercule") throw new Error("Bounded practice is only for hercule-v1.");
    this.training = training;
    this.maxMatches = maxMatches;
    this.botName = context.botName;
    this.strategy = context.strategy;
    this.policyVersion=context.policyVersion??(context.strategy==='hercule'?HERCULE_POLICY_VERSION:context.strategy==='victor'?VICTOR_POLICY_VERSION:SCORING_POLICY_VERSION);
    this.fixedProfile=context.fixedProfile;
    if(this.fixedProfile!==undefined&&(training||!(context.strategy==='defensive'&&['baseline','cautious','assertive','pressure'].includes(this.fixedProfile)||context.strategy==='hercule'&&this.fixedProfile==='hercule')))throw new Error('Fixed evaluation requires the matching policy without training.');
    this.schedule = context.strategy==='hercule'?['hercule']:training && process.env.BOT_TRAIN_PLAN
      ? process.env.BOT_TRAIN_PLAN.split(",").map((value) => value.trim() as Profile)
      : ["baseline", "assertive", "pressure", "pressure", "assertive", "baseline"];
    if (!this.schedule.length || this.schedule.some((value) => !(value in PROFILES) || value === "wags" || value === "victor" || value==='hercule'&&context.strategy!=='hercule')) throw new Error("Invalid training profile schedule.");
    const resume = training ? process.env.BOT_TRAIN_RESUME : undefined;
    if (resume && !/^[A-Za-z0-9_-]+$/.test(resume)) throw new Error("Invalid practice session name.");
    this.dir = join(root, resume ?? new Date().toISOString().replace(/[:.]/g, "-"));
    mkdirSync(this.dir, { recursive: true });
    if (resume) {
      const saved = JSON.parse(readFileSync(join(this.dir, "results.json"), "utf8"));
      if (saved.policyVersion!==this.policyVersion) throw new Error("Cannot resume a different policy version. Start a fresh practice session to keep comparisons valid.");
      this.results.push(...saved.results);
      for (const result of this.results) {
        this.completed.add(result.matchId);
        this.assignments.set(result.matchId, { profile: result.profile, phase: result.phase });
      }
      this.next = this.results.filter((result) => result.phase !== "partial").length;
      if (saved.plan) this.schedule = saved.plan;
    }
  }

  log(event: string, data: unknown): void {
    appendFileSync(join(this.dir, "events.jsonl"), JSON.stringify({ time: new Date().toISOString(), event, data },
      (_key, value) => typeof value === "bigint" ? value.toString() : value) + "\n");
  }

  assign(matchId: string, partial: boolean, normal: Profile = "cautious") {
    const prior = this.assignments.get(matchId);
    if (prior) return prior;
    const schedule = this.schedule;
    const assignment = {
      profile: this.strategy==='hercule'?'hercule':!this.training ? (this.strategy==="defensive"?this.fixedProfile??normal:this.strategy) : this.next < schedule.length ? schedule[this.next] : selectProfile(this.results),
      phase: partial ? "partial" : !this.training ? "normal" : this.strategy==='hercule'?'validation':this.next < schedule.length ? "exploration" : "validation",
    } as const;
    if (!partial) this.next++;
    this.assignments.set(matchId, assignment);
    this.log("match-start", { matchId, ...assignment });
    return assignment;
  }

  complete(result: MatchResult): boolean {
    if (this.completed.has(result.matchId)) return false;
    this.completed.add(result.matchId);
    this.results.push(result);
    this.log("match-end", result);
    this.write();
    return (this.training||this.fixedProfile!==undefined) && this.results.filter((item) => item.phase !== "partial").length >= this.maxMatches;
  }

  write(): void {
    const selected = this.strategy === "defensive" ? this.fixedProfile??selectProfile(this.results) : this.strategy;
    const data = { botName: this.botName, strategy: this.strategy, policyVersion:this.policyVersion,selected, weight: this.strategy==="victor"?null:PROFILES[selected],fixedEvaluation:this.fixedProfile!==undefined, plan: this.fixedProfile||this.strategy==="victor"||this.strategy==='hercule'?null:this.schedule,
      objective: this.strategy === "victor" ? "Build scoring income, protect a funded competitive position, and rebuild when needed; no profile tuning"
        :this.strategy === "wags" ? "Planned scoring and contender pressure; fixed offensive settings" :this.strategy==='hercule'?'Maximize own scoring income with productive public-word preferences; no defense or profile tuning':"0.5 * own score - highest real opponent score", results: this.results };
    writeFileSync(join(this.dir, "results.json"), JSON.stringify(data, null, 2));
    if(this.strategy==='hercule'){
      const rows=this.results.map(game=>{
        const real=game.opponents.filter(o=>!o.simulated),top=real.length?Math.max(...real.map(o=>o.score)):'none';
        return `| ${game.matchId} | ${game.phase} | ${game.score} | ${top} | ${game.rank}/${game.opponents.length+1} | ${game.words.length} | ${game.words.filter(w=>w.word.length>=7).length} | ${game.totalSpend} |`;
      });
      writeFileSync(join(this.dir,'REPORT.md'),[`# ${this.botName} offense-only results`,'',
        'Fixed productive acquisition and public-word preferences. No defensive bidding or profile tuning. These recorded outcomes do not establish superiority.','',
        '| Match | Phase | Score | Top real rival | Rank | Words | 7+ words | Letter spend |',
        '|---|---|---:|---:|---|---:|---:|---:|',...rows,'',
        `Policy: ${this.policyVersion}. Partial matches are recorded separately. Decision reasons and public goal preferences are recorded in events.jsonl.`,''].join('\n'));
      return;
    }
    if(this.strategy==="victor") {
      const rows=this.results.map(game=>{
        const real=game.opponents.filter(opponent=>!opponent.simulated);
        const top=real.length?Math.max(...real.map(opponent=>opponent.score)):"none";
        return `| ${game.matchId} | ${game.score} | ${top} | ${game.rank}/${game.opponents.length+1} | ${game.hybrid?.mode??"unknown"} | ${game.hybrid?.transitions??0} | ${game.defensiveWins} | ${game.totalSpend} |`;
      });
      writeFileSync(join(this.dir,"REPORT.md"),[
        `# ${this.botName} results`,"",
        "Adaptive build, defend, and rebuild strategy. These are recorded outcomes, not training results or proof of superiority.","",
        "| Match | Score | Top real rival | Rank | Final mode | Mode changes | Pressure purchases | Letter spend |",
        "|---|---:|---:|---|---|---:|---:|---:|",...rows,"",
        "Mode changes and their public-state reasons are recorded in events.jsonl. Pressure purchases do not prove a rival's points were denied.","",
      ].join("\n"));
      return;
    }
    if (this.strategy === "wags") {
      const rows = this.results.map((game) => {
        const real=game.opponents.filter(opponent=>!opponent.simulated);
        const top=real.length?Math.max(...real.map(opponent=>opponent.score)):"none";
        return `| ${game.matchId} | ${game.score} | ${top} | ${game.rank}/${game.opponents.length+1} | ${game.words.length} | ${game.words.filter((word)=>word.word.length<=4).length} | ${game.words.filter(word=>word.word.length>=7).length} | ${game.defensiveWins} | ${game.totalSpend} |`;
      });
      writeFileSync(join(this.dir,"REPORT.md"),[
        `# ${this.botName} results`, "", "Aggressive purchasing, planned word payouts, and contender pressure. These are recorded outcomes, not training results.", "",
        "| Match | Score | Top real rival | Rank | Words | Words of 2–4 letters | Words of 7+ letters | Pressure purchases | Letter spend |", "|---|---:|---:|---|---:|---:|---:|---:|---:|", ...rows, "",
        "Pressure purchases are wins after opponent pressure raised the bid. They do not prove that any particular points were denied.", "",
      ].join("\n"));
      return;
    }
    const rows = this.results.map((game) => {
      const real = game.opponents.filter((opponent) => !opponent.simulated);
      const max = real.length ? Math.max(...real.map((opponent) => opponent.score)) : "none";
      return `| ${game.matchId} | ${game.profile} | ${game.phase} | ${game.score} | ${max} | ${game.rank}/${game.opponents.length + 1} | ${real.length} | ${game.defensiveWins} | ${game.defensiveSpend} |`;
    });
    writeFileSync(join(this.dir, "REPORT.md"), [
      this.fixedProfile?"# hercule-v1 fixed-policy online test":"# hercule-v1 online defensive practice", "",
      "Matches run on the configured online game server. Simulated filler opponents are labeled and excluded from the tuning objective.", "",
      "| Match | Profile | Phase | Our score | Top real rival | Rank | Real rivals | Defense wins | Defense spend |",
      "|---|---|---|---:|---:|---|---:|---:|---:|", ...rows, "",
      this.fixedProfile?`Fixed tested policy: **${selected}**, weight ${PROFILES[selected]}. This run does not tune or change profiles.`:`Exploratory defensive choice: **${selected}**, weight ${PROFILES[selected]}. At least two eligible exploration games per candidate are required; otherwise the choice remains cautious.`, "",
      "Objective: 0.5 × our score − highest real opponent score. Higher is better. Validation games are not used to select a candidate.", "",
      "Defense wins are auctions won after a defensive reason raised our bid. They do not prove that the baseline would have lost the auction, or that a rival would have scored the estimated reward. Hidden starting racks are not guessed. Opponent and tile differences mean a small online batch cannot establish causal score suppression.", "",
    ].join("\n"));
  }
}
