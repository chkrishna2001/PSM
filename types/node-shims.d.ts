declare module "node:crypto" {
  export function randomUUID(): string;
}

declare module "node:fs" {
  export function readFileSync(path: string | number, encoding: BufferEncoding): string;
  export function writeFileSync(path: string, data: string, encoding?: BufferEncoding): void;
  export function appendFileSync(path: string, data: string, encoding?: BufferEncoding): void;
  export function existsSync(path: string): boolean;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function readdirSync(path: string): string[];
  export function renameSync(oldPath: string, newPath: string): void;
  export function unlinkSync(path: string): void;
  export function statSync(path: string): { size: number; mtimeMs: number; isDirectory(): boolean };
  export function createWriteStream(path: string): {
    write(chunk: Uint8Array): boolean;
    end(): void;
    once(event: "drain" | "finish" | "error", callback: (...args: unknown[]) => void): void;
  };
}

declare module "node:path" {
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
}

declare module "node:os" {
  export function homedir(): string;
  export function platform(): string;
}

declare module "node:test" {
  export default function test(name: string, fn: () => void | Promise<void>): void;
}

declare module "node:assert/strict" {
  const assert: {
    equal(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    ok(value: unknown, message?: string): void;
  };
  export default assert;
}

type BufferEncoding = "utf8" | "utf-8";

declare const process: {
  argv: string[];
  env: Record<string, string | undefined>;
  exitCode?: number;
  stdout: { write(data: string): void };
  stderr: { write(data: string): void };
};

declare function fetch(url: string, init?: {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
}): Promise<{
  ok: boolean;
  status: number;
  statusText: string;
  body?: unknown;
  json(): Promise<unknown>;
  text(): Promise<string>;
}>;

declare module "better-sqlite3" {
  namespace Database {
    interface Database {
      exec(sql: string): void;
      prepare(sql: string): Statement;
      close(): void;
    }

    interface Statement {
      run(...params: unknown[]): unknown;
      all(...params: unknown[]): Record<string, unknown>[];
      get(...params: unknown[]): Record<string, unknown> | undefined;
    }
  }

  interface DatabaseConstructor {
    new(path: string): Database.Database;
    (path: string): Database.Database;
  }

  const Database: DatabaseConstructor;
  export default Database;
}
