import { DatabaseSync } from "node:sqlite";

export type DbRow = Record<string, unknown>;

export interface SqliteStatement {
  run(...params: unknown[]): unknown;
  all(...params: unknown[]): DbRow[];
  get(...params: unknown[]): DbRow | undefined;
}

export interface SqliteDatabase {
  exec(sql: string): void;
  prepare(sql: string): SqliteStatement;
  close(): void;
  loadExtension?(path: string): void;
}

export function openSqliteDatabase(path: string): SqliteDatabase {
  return new NodeSqliteDatabase(path);
}

class NodeSqliteDatabase implements SqliteDatabase {
  private readonly db: DatabaseSync;

  constructor(path: string) {
    this.db = new DatabaseSync(path);
  }

  exec(sql: string): void {
    this.db.exec(sql);
  }

  prepare(sql: string): SqliteStatement {
    return this.db.prepare(sql);
  }

  close(): void {
    this.db.close();
  }

  loadExtension(path: string): void {
    this.db.loadExtension(path);
  }
}
