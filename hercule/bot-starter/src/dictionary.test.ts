import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, unlinkSync, rmdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { test } from "node:test";
import { loadDictionary } from "./dictionary.js";

test("loads the real wordlist on Windows and other platforms", () => {
  const dictionary = loadDictionary();
  assert.ok(dictionary.length > 100_000);
  assert.ok(dictionary.includes("QUIZ"));
});

test("handles spaces, CRLF, and lowercase; rejects an empty dictionary", () => {
  const directory = mkdtempSync(join(tmpdir(), "scrabblebot dictionary "));
  const source = join(directory, "my words.txt");
  try {
    writeFileSync(source, "cat\r\n AX \r\n\r\nnot-a-word\r\n");
    assert.deepEqual(loadDictionary(pathToFileURL(source)), ["CAT", "AX"]);
    writeFileSync(source, "\r\n");
    assert.throws(() => loadDictionary(pathToFileURL(source)), /empty/);
  } finally {
    unlinkSync(source);
    rmdirSync(directory);
  }
});
