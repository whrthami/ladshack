import re 

ATTRIBUTE_SIGNALS = {
    "budget": re.compile(r"\$\d+|\bunder\b|\bcheap|\bbudget|\bprice\b"),
    "brand": re.compile(r"\bnike|\badidas|\bsony|\bsamsung|\bapple\b"),  # extend as needed
    "size": re.compile(r"\bsize\s+\w+|\bsmall|\bmedium|\blarge|\bxl\b"),
    "color": re.compile(r"\bblack|\bred|\bblue|\bwhite|\bgreen|\bgrey|\bgray\b"),
    "category": None,  # handled separately: presence of any real term at all
}

def _resolve_ask_attribute(
    self, user_message: str, unique_terms: list[str], intent: str, confidence: float
) -> str | None:
    lowered = user_message.lower()

    # Too little signal overall -> we don't even know the category. Ask first.
    if len(unique_terms) <= 1:
        return "category"

    # Only worth asking on genuinely ambiguous turns — a confident Buying
    # query with a decent term count is probably specific enough already.
    if intent == "buying" and confidence > 0.6 and len(unique_terms) >= 3:
        return None

    # Check which useful attributes are still missing from THIS message.
    for attribute, pattern in ATTRIBUTE_SIGNALS.items():
        if pattern is None:
            continue
        if not pattern.search(lowered):
            return attribute  # ask about the first missing one

    return None  # message already covers the attributes we care about