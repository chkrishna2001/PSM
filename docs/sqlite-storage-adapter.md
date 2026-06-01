# SQLite Storage Adapter

## Summary

PSM Memory keeps SQLite access behind a small internal adapter so product memory logic is not coupled to a specific Node SQLite package.

Current default:

- `MemoryStore` depends on `SqliteDatabase` and `SqliteStatement` from `src/psm-core/src/sqlite.ts`.
- The default adapter uses Node's built-in `node:sqlite`.
- `better-sqlite3` is not a runtime dependency for the default install path.

This keeps company VM installs cleaner because SQLite storage no longer requires a native npm database package, `prebuild-install`, Python, or Windows C++ build tools.

## Why This Changed

The earlier SDK used `node:sqlite`, then moved to `better-sqlite3` during release packaging. Git history does not show a dedicated product rationale for that driver switch. Later company VM install testing showed `better-sqlite3` could fail on Node 23 when no prebuilt binary was available:

- npm fell back to `node-gyp rebuild`.
- `node-gyp` required Python and native build tools.
- locked-down company VMs could not complete that build.

The storage layer now defaults back to `node:sqlite` while keeping a driver boundary for future changes.

## Adapter Boundary

The store may use only this internal shape:

```ts
export interface SqliteStatement {
  run(...params: unknown[]): unknown;
  all(...params: unknown[]): Record<string, unknown>[];
  get(...params: unknown[]): Record<string, unknown> | undefined;
}

export interface SqliteDatabase {
  exec(sql: string): void;
  prepare(sql: string): SqliteStatement;
  close(): void;
  loadExtension?(path: string): void;
}
```

If PSM later needs `better-sqlite3`, `sqlite-vec`, or another driver, add or replace an adapter in `sqlite.ts`. Do not import third-party database drivers directly from memory store, service, ranking, hooks, or CLI code.

## Tradeoffs

- `node:sqlite` avoids the native npm install failure path for SQLite.
- Node currently prints an experimental SQLite warning on Node 22.
- `sqlite-vec` extension loading must be validated separately before relying on it.
- A future `better-sqlite3` adapter can be added for performance if install reliability is solved.
