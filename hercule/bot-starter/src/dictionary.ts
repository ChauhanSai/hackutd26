import { readFileSync } from "node:fs";

// File URLs work on Windows, including paths containing spaces.
export function loadDictionary(
  source = new URL("../../spacetimedb/wordlist.txt", import.meta.url),
): string[] {
  const words = readFileSync(source, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim().toUpperCase())
    .filter((word) => /^[A-Z]{2,}$/.test(word));
  if (words.length === 0) throw new Error("The word dictionary is empty.");
  return words;
}
