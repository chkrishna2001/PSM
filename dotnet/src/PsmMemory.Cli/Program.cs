using PsmMemory.Cli;

try
{
    return await CliRunner.RunAsync(args).ConfigureAwait(false);
}
catch (CliUsageException ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 1;
}
