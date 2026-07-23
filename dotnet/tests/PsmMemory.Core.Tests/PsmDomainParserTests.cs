using PsmMemory.Core.Runtime;
using Xunit;

namespace PsmMemory.Core.Tests;

/// <summary>
/// Covers the single shared domain-string parser that replaced three independently-diverged
/// copies (CLI direct commands, CLI hook commands, MCP tools). Strict mode preserves the
/// throw-on-bad-input behavior needed by the direct commands/MCP; LenientDefaultCoding preserves
/// the never-fail-the-agent's-turn behavior needed by hook commands.
/// </summary>
public class PsmDomainParserTests
{
    [Theory]
    [InlineData("coding", PsmDomain.Coding)]
    [InlineData("Coding", PsmDomain.Coding)]
    [InlineData("  coding  ", PsmDomain.Coding)]
    [InlineData("conversational", PsmDomain.Conversational)]
    [InlineData("CONVERSATIONAL", PsmDomain.Conversational)]
    [InlineData(" conversational ", PsmDomain.Conversational)]
    public void Parse_Strict_ValidValues_ReturnsExpectedDomain(string raw, PsmDomain expected)
    {
        var result = PsmDomainParser.Parse(raw, DomainParseMode.Strict);
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("bogus")]
    [InlineData("")]
    [InlineData("   ")]
    public void Parse_Strict_InvalidValue_Throws(string raw)
    {
        var ex = Assert.Throws<PsmDomainParseException>(() => PsmDomainParser.Parse(raw, DomainParseMode.Strict));
        Assert.Equal($"domain must be one of coding|conversational, got '{raw}'.", ex.Message);
    }

    [Fact]
    public void Parse_Strict_Null_Throws()
    {
        var ex = Assert.Throws<PsmDomainParseException>(() => PsmDomainParser.Parse(null, DomainParseMode.Strict));
        Assert.Equal("domain must be one of coding|conversational, got ''.", ex.Message);
    }

    [Theory]
    [InlineData("coding", PsmDomain.Coding)]
    [InlineData("Coding", PsmDomain.Coding)]
    [InlineData("conversational", PsmDomain.Conversational)]
    [InlineData("CONVERSATIONAL", PsmDomain.Conversational)]
    [InlineData("  conversational  ", PsmDomain.Conversational)]
    public void Parse_Lenient_ValidValues_ReturnsExpectedDomain(string raw, PsmDomain expected)
    {
        var result = PsmDomainParser.Parse(raw, DomainParseMode.LenientDefaultCoding);
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("bogus")]
    [InlineData("")]
    [InlineData("nope-not-a-domain")]
    public void Parse_Lenient_InvalidValue_DefaultsToCoding(string raw)
    {
        var result = PsmDomainParser.Parse(raw, DomainParseMode.LenientDefaultCoding);
        Assert.Equal(PsmDomain.Coding, result);
    }

    [Fact]
    public void Parse_Lenient_Null_DefaultsToCoding()
    {
        var result = PsmDomainParser.Parse(null, DomainParseMode.LenientDefaultCoding);
        Assert.Equal(PsmDomain.Coding, result);
    }
}
