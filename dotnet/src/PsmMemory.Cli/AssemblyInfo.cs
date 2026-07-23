using System.Runtime.CompilerServices;

// Lets PsmMemory.Cli.Tests unit-test the internal pure-logic helpers (HookIo, HookContextRenderer,
// HookCommands, InstallAgentCommand, ArgParser) directly instead of needing a real CLI process
// invocation for every test case.
[assembly: InternalsVisibleTo("PsmMemory.Cli.Tests")]
