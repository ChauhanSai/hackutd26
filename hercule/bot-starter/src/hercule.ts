import {herculeOffenseBid,chooseOffensiveHerculeWord} from './hercule-policy.js';
import {HERCULE_ACTIVE_CONFIG} from './hercule-settings.js';
import {type BidContext,type WordContext} from './strategy.js';
export {HERCULE_POLICY_VERSION} from './hercule-policy.js';
export {HERCULE_ACTIVE_CONFIG} from './hercule-settings.js';
export const decideHerculeBid=(ctx:BidContext)=>herculeOffenseBid(ctx,HERCULE_ACTIVE_CONFIG);
export const chooseHerculeWord=(ctx:WordContext)=>chooseOffensiveHerculeWord(ctx);
