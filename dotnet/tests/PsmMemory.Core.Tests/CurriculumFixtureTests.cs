using System.Text.Json;
using System.Text.Json.Nodes;
using PsmMemory.Core.Prompts;
using Xunit;

namespace PsmMemory.Core.Tests;

public class CurriculumFixtureTests
{
    private record CurriculumSourceRow(
        string Id,
        string Category,
        string Note,
        string LlmResponse,
        string[] ContextTurns,
        JsonNode AssistantDecision
    );

    private static string BuildCurriculumJsonLine(CurriculumSourceRow row)
    {
        var userText = PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns);
        
        // The prompt builder includes the system instruction and the assistant start marker.
        // We need to strip those to get just the user content for the "user" role, 
        // but actually, the prompt builder return is the WHOLE ChatML sequence.
        // Wait, the requirement says:
        // "Calls PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns) to get the user text."
        // But BuildStoragePrompt returns "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
        
        // Let's look at PromptBuilder.BuildStoragePrompt again:
        // return ChatMl(StorageSystemInstruction, user);
        // private static string ChatMl(string system, string user) => 
        //     $"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n";
        
        // The requirement B1 says: 
        // - Calls PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns) to get the user text.
        // - Builds the full JSON line with this exact shape:
        //   {"role":"system","content":"<PromptBuilder.StorageSystemInstruction, exactly>"},
        //   {"role":"user","content":"<the BuildStoragePrompt output>"}, ...
        
        // This is slightly contradictory because BuildStoragePrompt returns the whole ChatML block.
        // If I put the WHOLE BuildStoragePrompt output into the "user" content field, it will be double-wrapped.
        // However, looking at the seed.jsonl, the "user" content is ONLY the text:
        // "Extract durable memory... \n\nRecent context... \n\nAssistant response:\n..."
        
        // The PromptBuilder.BuildStoragePrompt method in the provided code:
        // public static string BuildStoragePrompt(string llmResponse, IReadOnlyList<string>? contextTurns = null)
        // {
        //     var user = StorageUserInstruction;
        //     if (contextTurns is { Count: > 0 }) { user += StorageContextLabel + string.Join("\n", contextTurns) + "\n\n"; }
        //     user += StorageResponseLabel + llmResponse.Trim();
        //     return ChatMl(StorageSystemInstruction, user);
        // }
        
        // It seems I need the 'user' part. But I can't modify PromptBuilder.
        // I can reverse the ChatML wrapping if I want, OR I can look at the requirement again:
        // "Calls PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns) to get the user text."
        
        // Actually, I can't "get the user text" from BuildStoragePrompt without stripping the ChatML.
        // BUT, the requirement also says: "Never hand-type or hardcode the rendered prompt string anywhere -- it must always come from actually calling PromptBuilder.BuildStoragePrompt at test time."
        
        // Wait, if I call BuildStoragePrompt, and it returns:
        // <|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n
        // and I want only {U}...
        
        // Let's check how I can extract {U} from that.
        // Or maybe the requirement meant "The result of the logic that BuildStoragePrompt uses".
        // But I can't change PromptBuilder.
        
        // Let's re-read: "Calls PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns) to get the user text."
        // Maybe the user text IS the whole thing? No, look at seed.jsonl:
        // "content":"Extract durable memory from the assistant response below.\nChoose ignore, store_episodic, or promote_semantic.\nWhen storing, emit grounded memory.content, facts[], and indexables[] from the text.\n\nRecent context, oldest first (for understanding only -- do NOT extract a memory from this section; base your decision only on the assistant response below):\nJolene: How old is Luna?\n\nAssistant response:\nDeborah: She is younger, she is 5 years old."
        
        // This is exactly the 'user' variable inside BuildStoragePrompt before it's passed to ChatMl.
        
        // Since I cannot change BuildStoragePrompt to return the raw user string, and I must call it,
        // I will call it and then strip the ChatML markers.
        // The markers are:
        // <|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n
        // and
        // <|im_end|>\n<|im_start|>assistant\n
        
        var fullPrompt = PromptBuilder.BuildStoragePrompt(row.LlmResponse, row.ContextTurns);
        
        // We know the structure: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n
        // To get {USER}, we find the first <|im_start|>user\n and the last <|im_end|> before <|im_start|>assistant\n.
        
        string userMarker = "<|im_start|>user\n";
        int start = fullPrompt.IndexOf(userMarker);
        if (start == -1) throw new Exception("User marker not found");
        start += userMarker.Length;
        
        int end = fullPrompt.LastIndexOf("<|im_end|>");
        if (end == -1) throw new Exception("End marker not found");
        
        string userContent = fullPrompt.Substring(start, end - start);

        var root = new JsonObject();
        root["id"] = JsonValue.Create(row.Id);
        root["category"] = JsonValue.Create(row.Category);
        root["note"] = JsonValue.Create(row.Note);
        
        var messages = new JsonArray();
        
        var systemMsg = new JsonObject();
        systemMsg["role"] = JsonValue.Create("system");
        systemMsg["content"] = JsonValue.Create(PromptBuilder.StorageSystemInstruction);
        messages.Add(systemMsg);
        
        var userMsg = new JsonObject();
        userMsg["role"] = JsonValue.Create("user");
        userMsg["content"] = JsonValue.Create(userContent);
        messages.Add(userMsg);
        
        var assistantMsg = new JsonObject();
        assistantMsg["role"] = JsonValue.Create("assistant");
        assistantMsg["content"] = JsonValue.Create(JsonSerializer.Serialize(row.AssistantDecision, new JsonSerializerOptions 
        { 
            WriteIndented = false,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        }));
        messages.Add(assistantMsg);

        root["messages"] = messages;

        return JsonSerializer.Serialize(root, new JsonSerializerOptions 
        { 
            WriteIndented = false,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });
    }

    [Fact]
    public void AllCurriculumSourceRows_MatchCommittedFixtureByteForByte()
    {
        // Resolve fixtures path.
        // The tests are in dotnet/tests/PsmMemory.Core.Tests/
        // The fixtures are in psm-model/prod-memory/fixtures/
        // Relative path: ../../../psm-model/prod-memory/fixtures/
        string fixturesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "psm-model", "prod-memory", "fixtures");
        
        // Actually, AppDomain.CurrentDomain.BaseDirectory is often where the DLL is (e.g. bin/Debug/net10.0)
        // Let's try to find the root of the repo. 
        // A common way in these projects is to use a fixed relative path from the executable.
        // Since I don't know the exact runtime path, I'll use a helper to find the project root.
        
        string projectRoot = FindProjectRoot();
        string sourcesPath = Path.Combine(projectRoot, "psm-model", "prod-memory", "fixtures", "context-aware-storage-curriculum-sources.jsonl");
        string seedPath = Path.Combine(projectRoot, "psm-model", "prod-memory", "fixtures", "context-aware-storage-seed.jsonl");

        var sourceLines = File.ReadAllLines(sourcesPath);
        var seedLines = File.ReadAllLines(seedPath);

        Assert.Equal(sourceLines.Length, seedLines.Length);

        for (int i = 0; i < sourceLines.Length; i++)
        {
            var sourceJson = JsonNode.Parse(sourceLines[i])!.AsObject();
            var row = new CurriculumSourceRow(
                sourceJson["id"]!.ToString(),
                sourceJson["category"]!.ToString(),
                sourceJson["note"]!.ToString(),
                sourceJson["llmResponse"]!.ToString(),
                sourceJson["contextTurns"]!.AsArray().Select(x => x!.ToString()).ToArray(),
                sourceJson["assistantDecision"]!
            );

            string generated = BuildCurriculumJsonLine(row);
            
            Assert.Equal(generated, seedLines[i]);
        }
    }

    private string FindProjectRoot()
    {
        string current = AppDomain.CurrentDomain.BaseDirectory;
        while (current != null)
        {
            if (Directory.Exists(Path.Combine(current, "psm-model")) && Directory.Exists(Path.Combine(current, "dotnet")))
            {
                return current;
            }
            current = Path.GetDirectoryName(current);
        }
        throw new Exception("Could not find project root");
    }
}
