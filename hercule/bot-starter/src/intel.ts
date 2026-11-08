export interface PublicRackEvent { time: bigint; kind: "purchase"|"word"; botId: string; text: string }
export interface Opponent {
  id: string;
  name: string;
  score: number;
  balance: number;
  isSimulated: boolean;
  knownRack: Map<string,number>;
  rackSize?: number;
  matchesPlayed?: number;
  wins?: number;
  rating?: number;
}
export interface RackObservation { knownRack: Map<string,number>; rackSize:number; exact:boolean }

export function observeRacks(events: PublicRackEvent[], startingRackSize: number, botIds: readonly string[]): Map<string,RackObservation> {
  const observations=new Map(botIds.map(id=>[id,{knownRack:new Map<string,number>(),rackSize:startingRackSize,exact:startingRackSize===0}]));
  const ordered=[...events].sort((a,b)=>a.time<b.time?-1:a.time>b.time?1:a.kind===b.kind?0:a.kind==="purchase"?-1:1);
  for (const event of ordered) {
    let observation=observations.get(event.botId);
    if (!observation) {observation={knownRack:new Map(),rackSize:startingRackSize,exact:startingRackSize===0};observations.set(event.botId,observation);}
    for (const letter of event.text) {
      const have=observation.knownRack.get(letter)??0;
      observation.knownRack.set(letter,event.kind==="purchase"?have+1:Math.max(0,have-1));
      observation.rackSize+=event.kind==="purchase"?1:-1;
    }
    // Negative sizes indicate incomplete history, never an exact empty rack.
    observation.exact=observation.rackSize>=0 && [...observation.knownRack.values()].reduce((s,n)=>s+n,0)===observation.rackSize;
  }
  return observations;
}

export function knownRacks(events: PublicRackEvent[]): Map<string,Map<string,number>> {
  return new Map([...observeRacks(events,5,[])].map(([id,observation])=>[id,observation.knownRack]));
}

// Public-history priors only, collected 2026-10-24 16:30 UTC. Live leaderboard
// rows supersede these. They rank threats; they never supply private letters.
export const CONTENDER_PRIORS: Readonly<Record<string,{wins:number;matchesPlayed:number;rating:number}>>={
  Nijello:{wins:146,matchesPlayed:372,rating:1558},
  "Chinese Dante":{wins:194,matchesPlayed:536,rating:1612},
  LetterShark:{wins:66,matchesPlayed:188,rating:1234},
  YeziShanYang:{wins:69,matchesPlayed:229,rating:1263},
  Doppio:{wins:124,matchesPlayed:412,rating:1530},
  Metheny:{wins:24,matchesPlayed:82,rating:1660},
  "Lambi Lambiii":{wins:31,matchesPlayed:107,rating:1628},
  Karthik:{wins:39,matchesPlayed:148,rating:1105},
  keanu_reeves:{wins:84,matchesPlayed:340,rating:1602},
  guesser:{wins:103,matchesPlayed:427,rating:1419},
  benx:{wins:62,matchesPlayed:258,rating:1261},
  postturing:{wins:34,matchesPlayed:142,rating:1636},
  bro:{wins:25,matchesPlayed:321,rating:1617},
};

export function contenderStrength(opponent: Opponent): number {
  const prior=CONTENDER_PRIORS[opponent.name];
  const games=opponent.matchesPlayed??prior?.matchesPlayed??0;
  const wins=opponent.wins??prior?.wins??0;
  const rating=opponent.rating??prior?.rating??1200;
  const reliability=games/(games+30);
  return 1+reliability*(Math.min(0.6,Math.max(0,wins/Math.max(1,games)-1/6)*2)+Math.min(0.3,Math.max(0,rating-1400)/1000));
}
