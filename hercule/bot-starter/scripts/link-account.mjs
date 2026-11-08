import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";

if (process.argv.includes("--help")) {
  console.log("Open https://launchpad-scrabblebot.vercel.app/account, then run Link-Account.cmd.");
  console.log("Paste the public Web identity from that page when prompted. Sign in when the browser opens.");
  process.exit(0);
}

const root = fileURLToPath(new URL("../../", import.meta.url));
const cli = join(root, ".tools", "spacetime", "spacetimedb-cli.exe");
const rootDir = join(root, ".tools", "competition-cli");
console.log("Open https://launchpad-scrabblebot.vercel.app/account.");
console.log("Copy the public Web identity displayed on that page (not a bot token).");
const terminal = createInterface({ input: process.stdin, output: process.stdout });
const identity = (await terminal.question("Web identity: ")).trim().replace(/^0x/i, "");
terminal.close();
if (!/^[0-9a-f]{64}$/i.test(identity)) {
  console.error("Expected a 64-character hexadecimal Web identity. Copy it from the account page.");
  process.exit(1);
}

function run(args) {
  const result = spawnSync(cli, ["--root-dir", rootDir, ...args], {
    cwd: root, stdio: "inherit",
  });
  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
}

// Login belongs to the human; the website separately creates the bot token.
run(["login"]);
run([
  "call", "--server", "https://maincloud.spacetimedb.com", "--no-config", "--yes",
  "scrabblebot", "connect_id", JSON.stringify(`0x${identity}`),
]);
console.log("Account linked. Refresh the account page, create or join a team, then Generate token on /team.");
console.log("Run Start-Bot.cmd and paste that bot token into its hidden prompt.");
