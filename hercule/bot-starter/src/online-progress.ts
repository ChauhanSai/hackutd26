export const ONLINE_ADAPTER_VERSION='remaining-bag-v2';
// The server decrements bagTotal for both the initial deal and every auction
// draw. During round N it already excludes the tile currently being auctioned.
export function auctionProgress(match:{currentRound:number;bagTotal:number}) {
  const remainingTiles=Math.max(0,match.bagTotal);
  const auctions=Math.max(1,match.currentRound+remainingTiles);
  return {remainingTiles,progress:Math.min(1,Math.max(0,(match.currentRound-1)/Math.max(1,auctions-1)))};
}
