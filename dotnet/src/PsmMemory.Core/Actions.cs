namespace PsmMemory.Core;

/// <summary>
/// Ported directly from psm-core/src/actions.ts. Normalizes free-form model action strings into
/// the canonical MemoryAction vocabulary, and maps each action to the store route that handles it.
/// </summary>
public static class Actions
{
    /// <summary>Canonical action vocabulary (mirrors types.ts MemoryAction union).</summary>
    public static class Kinds
    {
        public const string Ignore = "ignore";
        public const string Store = "store";
        public const string StoreEpisodic = "store_episodic";
        public const string Promote = "promote";
        public const string PromoteSemantic = "promote_semantic";
        public const string Update = "update";
        public const string UpdateExisting = "update_existing";
        public const string Rank = "rank";
        public const string Decay = "decay";
        public const string DecayAndUpdate = "decay_and_update";
        public const string FlagConflict = "flag_conflict";
        public const string FlagAndStore = "flag_and_store";
        public const string FlagAndUpdate = "flag_and_update";
        public const string DetectInterference = "detect_interference";
    }

    /// <summary>Store routes (mirrors types.ts MemoryRoute union).</summary>
    public static class Routes
    {
        public const string Ignore = "ignore";
        public const string RecallOnly = "recall_only";
        public const string EpisodicInsert = "episodic_insert";
        public const string SemanticUpsert = "semantic_upsert";
        public const string UpdateWithSupersede = "update_with_supersede";
        public const string DecayExistingThenInsert = "decay_existing_then_insert";
        public const string ConflictLogAndHold = "conflict_log_and_hold";
    }

    public static string NormalizeAction(string? action)
    {
        var value = (action ?? string.Empty).Trim().ToLowerInvariant();
        return value switch
        {
            "ignore" or "ignore_noise" => Kinds.Ignore,
            "store" => Kinds.Store,
            "store_episodic" => Kinds.StoreEpisodic,
            "store_semantic" => Kinds.PromoteSemantic,
            "promote" => Kinds.Promote,
            "promote_semantic" => Kinds.PromoteSemantic,
            "update" => Kinds.Update,
            "update_existing" => Kinds.UpdateExisting,
            "rank" or "recall_weighting" => Kinds.Rank,
            "decay" => Kinds.Decay,
            "decay_and_update" => Kinds.DecayAndUpdate,
            "flag_conflict" => Kinds.FlagConflict,
            "flag_and_store" => Kinds.FlagAndStore,
            "flag_and_update" => Kinds.FlagAndUpdate,
            "detect_interference" => Kinds.DetectInterference,
            _ => Kinds.StoreEpisodic
        };
    }

    public static string RouteForAction(string action) => action switch
    {
        Kinds.Ignore => Routes.Ignore,
        Kinds.Rank => Routes.RecallOnly,
        Kinds.Promote or Kinds.PromoteSemantic => Routes.SemanticUpsert,
        Kinds.Update or Kinds.UpdateExisting => Routes.UpdateWithSupersede,
        Kinds.Decay or Kinds.DecayAndUpdate => Routes.DecayExistingThenInsert,
        Kinds.FlagConflict or Kinds.FlagAndStore or Kinds.FlagAndUpdate or Kinds.DetectInterference => Routes.ConflictLogAndHold,
        _ => Routes.EpisodicInsert
    };
}
