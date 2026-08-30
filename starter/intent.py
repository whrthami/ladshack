import re
from agent import _terms

BUYING_WORDS = {
    "buy", "purchase", "order", "checkout", "cart", "ship", "shipping",
    "deliver", "delivery", "cheapest", "price", "discount", "deal",
    "available", "stock", "reorder", "gift",
}
BROWSING_WORDS = {
    "want", "looking", "would", "please", "options", "recommend",
    "suggest", "compare", "comparison", "difference", "best", "good",
    "explore", "browse", "curious", "thinking", "maybe", "considering",
}

BUYING_PATTERNS = [
    re.compile(r"\bunder\s*\$?\d+\b"),          # "under $50"
    re.compile(r"\bsize\s+\w+\b"),               # "size L", "size 10"
    re.compile(r"\bin\s+(black|red|blue|white|green|grey|gray)\b"),
    re.compile(r"\b\d+\s*(gb|tb|inch|in|ml|oz|lb)\b"),  # spec constraints
    re.compile(r"\bfor\s+my\b"),                 # "for my son", often a decided gift purchase
]
BROWSING_PATTERNS = [
    re.compile(r"\bwhat('?s| is)\b.*\bgood for\b"),
    re.compile(r"\bdifference between\b"),
    re.compile(r"\boptions? for\b"),
    re.compile(r"\bwhat.*(recommend|suggest)\b"),
    re.compile(r"\bwhich\b.*\bbetter\b"),
]


def _detect_intent(text: str) -> tuple[str, float]:
    """Classify a message as 'buying' or 'browsing'. Returns (label, confidence)."""
    lowered = text.lower()
    tokens = set(_terms(text))  # reuses existing tokenizer, minus filler stopwords

    buy_score = len(tokens & BUYING_WORDS)
    browse_score = len(tokens & BROWSING_WORDS)

    buy_score += sum(2 for pattern in BUYING_PATTERNS if pattern.search(lowered))
    browse_score += sum(2 for pattern in BROWSING_PATTERNS if pattern.search(lowered))

    total = buy_score + browse_score
    if total == 0:
        return "browsing", 0.0  # default to the wider, safer net when signal is absent

    if buy_score >= browse_score:
        return "buying", buy_score / total
    return "browsing", browse_score / total

def _resolve_top_k(self, intent: str, confidence: float, top_k: int) -> int:
    """Buying + confident -> tighter results. Browsing/ambiguous -> full width."""
    if intent == "buying" and confidence > 0.6:
        return min(top_k, 3)
    return top_k

def _resolve_weights(self, intent: str, confidence: float) -> str:
    """Return a comma-separated bm25() weight string, column order:
    parent_asin, title, categories, features, details, store, description
    """
    if intent == "buying" and confidence > 0.6:
        # Buyers use precise language -> trust title/features/details more
        return "0.0, 6.0, 3.0, 4.0, 4.0, 1.5, 0.5"
    # Browsing -> trust category/description more for exploratory matching
    return "0.0, 5.0, 5.0, 2.0, 2.0, 1.5, 2.5"

def _resolve_message(self, intent: str, recommendations: list[dict]) -> str:
    if not recommendations:
        return "I couldn't find a close match — want to tell me more about what you need?"
    if intent == "buying":
        return "Here are the closest matches to buy."
    return "Here are a few options to explore."