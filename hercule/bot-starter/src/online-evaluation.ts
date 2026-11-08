export function onlineEvaluation(env:Record<string,string|undefined>) {
  if(env.BOT_EVAL_MATCHES===undefined)return undefined;
  const matches=Number(env.BOT_EVAL_MATCHES),minutes=Number(env.BOT_EVAL_MINUTES??90);
  if(!/^\d+$/.test(env.BOT_EVAL_MATCHES)||!Number.isInteger(matches)||matches<1||matches>50)throw new Error('Online test matches must be between 1 and 50.');
  if(!Number.isInteger(minutes)||minutes<1||minutes>180)throw new Error('Online test time limit must be between 1 and 180 minutes.');
  if(!['hercule','defensive'].includes(env.BOT_STRATEGY??'hercule')||(env.BOT_NAME??'hercule-v1')!=='hercule-v1')throw new Error('The fixed-policy online test uses hercule-v1.');
  if(env.BOT_TRAIN==='1'||env.TOURNAMENT_ID)throw new Error('Fixed-policy tests cannot tune profiles or register for tournaments.');
  return {matches,minutes,profile:'hercule' as const,weight:0};
}
export function assertBotIdentity(expected:string,registered:string|undefined) {
  if(registered!==expected)throw new Error(`Expected ${expected}'s credential; the connected bot is ${registered??'unknown'}. No matches entered.`);
}
