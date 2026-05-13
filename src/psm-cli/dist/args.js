export function parseArgs(argv) {
    const [command = "help", ...rest] = argv;
    const options = {};
    for (let i = 0; i < rest.length; i++) {
        const token = rest[i];
        if (!token.startsWith("--"))
            continue;
        const key = token.slice(2);
        const next = rest[i + 1];
        if (next && !next.startsWith("--")) {
            options[key] = next;
            i++;
        }
        else {
            options[key] = true;
        }
    }
    return { command: command.toLowerCase(), options };
}
export function required(options, key) {
    const value = options[key];
    if (typeof value === "string" && value.trim())
        return value;
    throw new Error(`Missing required option --${key}`);
}
export function stringOption(options, key, fallback) {
    const value = options[key];
    return typeof value === "string" && value.trim() ? value : fallback;
}
export function intOption(options, key, fallback) {
    const value = options[key];
    const parsed = typeof value === "string" ? Number(value) : Number.NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
export function boolOption(options, key) {
    return options[key] === true || options[key] === "true";
}
//# sourceMappingURL=args.js.map