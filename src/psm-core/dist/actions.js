export function normalizeAction(action) {
    const value = String(action ?? "").trim().toLowerCase();
    switch (value) {
        case "ignore":
        case "ignore_noise":
            return "ignore";
        case "store":
            return "store";
        case "store_episodic":
            return "store_episodic";
        case "store_semantic":
            return "promote_semantic";
        case "promote":
            return "promote";
        case "promote_semantic":
            return "promote_semantic";
        case "update":
            return "update";
        case "update_existing":
            return "update_existing";
        case "rank":
        case "recall_weighting":
            return "rank";
        case "decay":
            return "decay";
        case "decay_and_update":
            return "decay_and_update";
        case "flag_conflict":
            return "flag_conflict";
        case "flag_and_store":
            return "flag_and_store";
        case "flag_and_update":
            return "flag_and_update";
        case "detect_interference":
            return "detect_interference";
        default:
            return "store_episodic";
    }
}
export function routeForAction(action) {
    switch (action) {
        case "ignore":
            return "ignore";
        case "rank":
            return "recall_only";
        case "promote":
        case "promote_semantic":
            return "semantic_upsert";
        case "update":
        case "update_existing":
            return "update_with_supersede";
        case "decay":
        case "decay_and_update":
            return "decay_existing_then_insert";
        case "flag_conflict":
        case "flag_and_store":
        case "flag_and_update":
        case "detect_interference":
            return "conflict_log_and_hold";
        default:
            return "episodic_insert";
    }
}
export function actionFromOperation(operation) {
    const value = operation.toLowerCase();
    if (value.includes("ignore"))
        return "ignore";
    if (value.includes("interference") || value.includes("conflict"))
        return "detect_interference";
    if (value.includes("promote"))
        return "promote";
    if (value.includes("update"))
        return "update_existing";
    if (value.includes("rank") || value.includes("recall"))
        return "rank";
    if (value.includes("decay"))
        return "decay";
    return "store_episodic";
}
//# sourceMappingURL=actions.js.map