import {type HerculeConfig} from './hercule-policy.js';
// Selected on development bags only. The subsequent validation is reported in
// HERCULE-OFFENSE-RESULTS.md; public-word preferences remain a small heuristic.
export const HERCULE_ACTIVE_CONFIG:Readonly<HerculeConfig>={id:'public-six',mode:'public',maxExtra:6};
