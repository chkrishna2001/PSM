using PsmMemory.Core.Prompts;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Locks down PromptBuilder.BuildStoragePrompt's exact text, with particular emphasis on the
/// no-context path staying byte-identical to before the 2026-07-23 context-window parameter was
/// added -- the currently-running production adapters were trained on that exact string and have
/// never seen a context block, so any accidental drift here would silently degrade inference
/// without any adapter retraining involved.
/// </summary>
public class PromptBuilderTests
{
    private const string ExpectedNoContextPrompt =
        "<|im_start|>system\n" +
        "You are the PSM storage model.\n" +
        "Return one strict JSON object compatible with the PSM StorageDecision schema.\n" +
        "First write reasoning explaining your decision, then action, memory (or null),\n" +
        "facts[], and indexables[] in that order.\n" +
        "Do not include markdown, prose, comments, or fallback text outside JSON.\n" +
        "Facts must be explicit and supported by evidence_text from the current input.<|im_end|>\n" +
        "<|im_start|>user\n" +
        "Extract durable memory from the assistant response below.\n" +
        "Choose ignore, store_episodic, or promote_semantic.\n" +
        "When storing, emit grounded memory.content, facts[], and indexables[] from the text.\n\n" +
        "Assistant response:\n" +
        "The user's favorite editor is Neovim.<|im_end|>\n" +
        "<|im_start|>assistant\n";

    [Fact]
    public void BuildStoragePrompt_WithNoContextArgument_MatchesExactPreExistingFormat()
    {
        var actual = PromptBuilder.BuildStoragePrompt("The user's favorite editor is Neovim.");
        Assert.Equal(ExpectedNoContextPrompt, actual);
    }

    [Fact]
    public void BuildStoragePrompt_WithNullContext_IsByteIdenticalToOmittedArgument()
    {
        var withoutArg = PromptBuilder.BuildStoragePrompt("The user's favorite editor is Neovim.");
        var withNull = PromptBuilder.BuildStoragePrompt("The user's favorite editor is Neovim.", null);
        Assert.Equal(withoutArg, withNull);
    }

    [Fact]
    public void BuildStoragePrompt_WithEmptyContextList_IsByteIdenticalToOmittedArgument()
    {
        var withoutArg = PromptBuilder.BuildStoragePrompt("The user's favorite editor is Neovim.");
        var withEmpty = PromptBuilder.BuildStoragePrompt("The user's favorite editor is Neovim.", Array.Empty<string>());
        Assert.Equal(withoutArg, withEmpty);
    }

    [Fact]
    public void BuildStoragePrompt_WithContext_InsertsLabeledBlockBeforeAssistantResponse()
    {
        var actual = PromptBuilder.BuildStoragePrompt(
            "It's Shia Labeouf!",
            new[] { "Jon: Sounds familiar, who do those words belong to?" });

        Assert.Contains(
            "Recent context, oldest first (for understanding only -- do NOT extract a memory from this " +
            "section; base your decision only on the assistant response below):\n" +
            "Jon: Sounds familiar, who do those words belong to?\n\n" +
            "Assistant response:\n" +
            "It's Shia Labeouf!",
            actual);
    }

    [Fact]
    public void BuildStoragePrompt_WithMultipleContextTurns_JoinsThemInOrderOldestFirst()
    {
        var actual = PromptBuilder.BuildStoragePrompt(
            "It's Shia Labeouf!",
            new[]
            {
                "Gina: Remember, just do it!",
                "Jon: Sounds familiar, who do those words belong to?"
            });

        Assert.Contains(
            "Gina: Remember, just do it!\n" +
            "Jon: Sounds familiar, who do those words belong to?\n\n" +
            "Assistant response:\n" +
            "It's Shia Labeouf!",
            actual);
    }

    [Fact]
    public void BuildStorageRepairPrompt_Unaffected_MatchesPreExistingFormat()
    {
        var actual = PromptBuilder.BuildStorageRepairPrompt("some response", "not json");
        Assert.Contains("Assistant response:\nsome response", actual);
        Assert.Contains("Your previous answer was not valid JSON", actual);
        Assert.DoesNotContain("Recent context", actual);
    }
}

public class PromptBuilderCurriculumConsistencyTests
{
    [Fact]
    public void BuildStoragePrompt_MatchesHandWrittenCurriculumSample_LunaAgeExample()
    {
        var prompt = PromptBuilder.BuildStoragePrompt(
            "Deborah: She is younger, she is 5 years old.",
            new[] { "Jolene: How old is Luna?" });

        var expectedUser =
            "Extract durable memory from the assistant response below.\n" +
            "Choose ignore, store_episodic, or promote_semantic.\n" +
            "When storing, emit grounded memory.content, facts[], and indexables[] from the text.\n\n" +
            "Recent context, oldest first (for understanding only -- do NOT extract a memory from this section; base your decision only on the assistant response below):\n" +
            "Jolene: How old is Luna?\n\n" +
            "Assistant response:\n" +
            "Deborah: She is younger, she is 5 years old.";

        Assert.Contains(expectedUser, prompt);
    }
}
