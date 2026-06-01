declare module "node:crypto" {
  export function randomUUID(): string;
}

declare module "node:fs" {
  export function readFileSync(path: string | number, encoding: BufferEncoding): string;
  export function writeFileSync(path: string, data: string, encoding?: BufferEncoding): void;
  export function appendFileSync(path: string, data: string, encoding?: BufferEncoding): void;
  export function copyFileSync(src: string, dest: string): void;
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

declare module "node:http" {
  export interface IncomingMessage {
    method?: string;
    url?: string;
    on(event: "data" | "end" | "error", callback: (...args: unknown[]) => void): void;
  }

  export interface ServerResponse {
    statusCode: number;
    setHeader(name: string, value: string): void;
    end(data: string): void;
  }

  export interface ClientResponse {
    statusCode?: number;
    setEncoding(encoding: BufferEncoding): void;
    on(event: "data" | "end", callback: (...args: unknown[]) => void): void;
  }

  export interface Server {
    listen(port: number, host: string, callback: () => void): void;
    once(event: "error", callback: (error: unknown) => void): void;
    close(callback: () => void): void;
    address(): string | { port?: number } | null;
  }

  export interface ClientRequest {
    once(event: "error", callback: (error: unknown) => void): void;
    write(data: string): void;
    end(): void;
  }

  export function createServer(callback: (req: IncomingMessage, res: ServerResponse) => void): Server;
  export function request(options: {
    hostname: string;
    port: number;
    path: string;
    method: string;
    headers?: Record<string, string>;
  }, callback: (res: ClientResponse) => void): ClientRequest;
}

declare module "node:child_process" {
  export interface ChildProcess {
    stdin?: { write(data: string, encoding?: BufferEncoding, callback?: (error?: Error | null) => void): void; end(): void };
    stdout?: { on(event: "data", callback: (chunk: unknown) => void): void };
    stderr?: { on(event: "data", callback: (chunk: unknown) => void): void };
    on(event: "error", callback: (error: Error) => void): void;
    on(event: "exit", callback: (code: number | null, signal: string | null) => void): void;
    unref(): void;
  }
  export function spawn(command: string, args?: string[], options?: {
    detached?: boolean;
    stdio?: "ignore" | Array<"ignore" | "pipe">;
    windowsHide?: boolean;
  }): ChildProcess;
  export function spawnSync(command: string, args?: string[], options?: {
    cwd?: string;
    encoding?: BufferEncoding;
    stdio?: Array<"ignore" | "pipe">;
  }): {
    status: number | null;
    stdout: string;
    stderr: string;
  };
}

declare module "node:path" {
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
  export function resolve(...paths: string[]): string;
}

declare module "node:url" {
  export function fileURLToPath(url: URL | string): string;
}

declare module "node:os" {
  export function homedir(): string;
  export function platform(): string;
  export function userInfo(): { username: string };
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

declare module "node:readline" {
  export function createInterface(options: {
    input: unknown;
    output?: unknown;
  }): {
    on(event: "line", callback: (line: string) => void): void;
    question(query: string, callback: (answer: string) => void): void;
    close(): void;
  };
}

declare module "node:sqlite" {
  export class DatabaseSync {
    constructor(path: string);
    exec(sql: string): void;
    prepare(sql: string): {
      run(...params: unknown[]): unknown;
      all(...params: unknown[]): Record<string, unknown>[];
      get(...params: unknown[]): Record<string, unknown> | undefined;
    };
    close(): void;
    loadExtension(path: string): void;
  }
}

type BufferEncoding = "utf8" | "utf-8";

declare const process: {
  argv: string[];
  execPath: string;
  env: Record<string, string | undefined>;
  exitCode?: number;
  pid: number;
  cwd(): string;
  stdin: { isTTY?: boolean };
  stdout: { write(data: string): void; isTTY?: boolean };
  stderr: { write(data: string): void };
  exit(code?: number): never;
  once(event: "SIGINT" | "SIGTERM", callback: () => void): void;
};

declare const Buffer: {
  byteLength(value: string): number;
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
