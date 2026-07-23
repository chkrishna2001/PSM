using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using PsmMemory.Core;

namespace PsmMemory.Mcp;

/// <summary>
/// Background host service that continuously drains the fire-and-forget remember() queue (see
/// <see cref="RememberQueueDrainer"/>) for the lifetime of this MCP server process. Polls every
/// <see cref="PollInterval"/>; when a full batch was drained (there's likely more backlog), it loops
/// immediately instead of sleeping, to clear a backlog faster.
/// </summary>
public sealed class RememberQueueWorker : BackgroundService
{
    private static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(5);
    private const int BatchSize = 10;

    private readonly RememberQueueDrainer _drainer;
    private readonly ILogger<RememberQueueWorker> _logger;

    public RememberQueueWorker(RememberQueueDrainer drainer, ILogger<RememberQueueWorker> logger)
    {
        _drainer = drainer;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var processed = await _drainer.DrainOnceAsync(BatchSize, stoppingToken).ConfigureAwait(false);
                if (processed < BatchSize)
                {
                    await Task.Delay(PollInterval, stoppingToken).ConfigureAwait(false);
                }
                // else: batch was full, loop again immediately to keep draining the backlog.
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "RememberQueueWorker: unhandled error draining the remember queue; retrying after the poll interval.");
                await Task.Delay(PollInterval, stoppingToken).ConfigureAwait(false);
            }
        }
    }
}
