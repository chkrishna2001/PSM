export interface ParsedArgs {
    command: string;
    options: Record<string, string | boolean>;
}
export declare function parseArgs(argv: string[]): ParsedArgs;
export declare function required(options: Record<string, string | boolean>, key: string): string;
export declare function stringOption(options: Record<string, string | boolean>, key: string, fallback: string): string;
export declare function intOption(options: Record<string, string | boolean>, key: string, fallback: number): number;
export declare function boolOption(options: Record<string, string | boolean>, key: string): boolean;
