#!/usr/bin/env node
import { existsSync } from "node:fs";
import Database from "better-sqlite3";

const paths = process.argv.slice(2);
if (paths.length === 0) {
  console.error("Usage: node nano-psm/data-pipeline/src/inspect-local-psm-sources.mjs <db-path> [...]");
  process.exit(2);
}

const results = paths.map(inspectDb);
console.log(JSON.stringify({ checked: results.length, sources: results }, null, 2));

function inspectDb(path) {
  if (!existsSync(path)) {
    return { path, exists: false, episodic: 0, semantic: 0, memory_facts: 0, decisions: 0 };
  }
  const db = new Database(path, { readonly: true, fileMustExist: true });
  try {
    return {
      path,
      exists: true,
      episodic: countTable(db, "episodic"),
      semantic: countTable(db, "semantic"),
      memory_facts: countTable(db, "memory_facts"),
      decisions: countTable(db, "decisions")
    };
  } finally {
    db.close();
  }
}

function countTable(db, table) {
  const exists = db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(table);
  return exists ? db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count : 0;
}

