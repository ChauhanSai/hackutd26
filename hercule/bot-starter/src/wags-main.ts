// Separate local name, token file, strategy, and logs; importing starts the runner.
process.env.BOT_NAME = "Wags";
process.env.BOT_STRATEGY = "wags";
await import("./online.js");
export {};
