from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
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

SLOT_PATTERNS = {
        "budget": re.compile(r"(?:under|below|less than)\s*\$?(\d+)|\$(\d+)"),
        "brand": re.compile(
            r"\b(nike|adidas|sony|samsung|apple|puma|reebok|lg|hp|dell|lenovo)\b",
            re.IGNORECASE,
        ),
        "size": re.compile(r"\bsize\s+(\w+)\b|\b(small|medium|large|xl|xs|xxl)\b", re.IGNORECASE),
        "color": re.compile(
            r"\b(black|red|blue|white|green|grey|gray|yellow|pink|purple|orange|brown)\b",
            re.IGNORECASE,
        ),
    }
FULL_WEIGHT = 1.0
DECAY_RATE = 0.85
WARM_THRESHOLD = 0.3  # slot still "worth trusting" above this weight

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

def _detect_intent(text: str) -> tuple[str, float]:
    """Classify a message as 'buying' or 'browsing'. Returns (label, confidence)."""
    lowered = text.lower()
    tokens = set(TOKEN_RE.findall(lowered)) 

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

class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._category_vocab: set[str] = set()  
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                categories_text = _text(product.get("categories"))
                self._category_vocab.update(_terms(categories_text)) 
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        categories_text,
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # This just initializes an EMPTY memory structure for this session.
        self._sessions[session_id] = {
            "slots": {
                "category": {"value": None, "weight": 0.0},
                "budget": {"value": None, "weight": 0.0},
                "brand": {"value": None, "weight": 0.0},
                "size": {"value": None, "weight": 0.0},
                "color": {"value": None, "weight": 0.0},
            },
            "history": [],
            "rejected_asins": set(),
            "turn": 0,
        }

    def _extract_slots(self, user_message: str, slots: dict) -> None:
        lowered = user_message.lower()
        matched_values: set[str] = set()

        for name, pattern in self.SLOT_PATTERNS.items():
            match = pattern.search(lowered)
            if match:
                value = next((g for g in match.groups() if g), None)
                if value:
                    slots[name] = {"value": value, "weight": self.FULL_WEIGHT}
                    matched_values.add(value.lower())

        # Only consider terms that actually appear in real catalog categories
        remaining_terms = [
            t for t in _terms(user_message)
            if t not in matched_values and t in self._category_vocab
        ]
        if remaining_terms:
            candidate = max(remaining_terms, key=len)
            slots["category"] = {"value": candidate, "weight": self.FULL_WEIGHT}

    def _resolve_weights(self, intent: str, confidence: float, slots: dict) -> str:
            """Column order: parent_asin, title, categories, features, details, store, description"""
            if intent == "buying" and confidence > 0.6:
                base = [0.0, 6.0, 3.0, 4.0, 4.0, 1.5, 0.5]
            else:
                base = [0.0, 5.0, 5.0, 2.0, 2.0, 1.5, 2.5]

            # NEW: nudge store weight if brand memory is still warm, even if
            # this turn's message doesn't mention a brand at all
            brand_weight = slots.get("brand", {}).get("weight", 0.0)
            if brand_weight > self.WARM_THRESHOLD:
                base[5] += 2.0 * brand_weight  # index 5 = store column

            # NEW: nudge features/details weight if size/color memory is warm —
            # these tend to show up in features/details text more than title
            attribute_weight = max(
                slots.get("size", {}).get("weight", 0.0),
                slots.get("color", {}).get("weight", 0.0),
            )
            if attribute_weight > self.WARM_THRESHOLD:
                base[3] += 1.5 * attribute_weight  # features
                base[4] += 1.5 * attribute_weight  # details

            return ", ".join(str(w) for w in base)


    def _resolve_top_k(self, intent: str, confidence: float, top_k: int) -> int:
        """Buying + confident -> tighter results. Browsing/ambiguous -> full width."""
        if intent == "buying" and confidence > 0.6:
            return min(top_k, 3)
        return top_k

    def _resolve_ask_attribute(
        self, user_message: str, unique_terms: list[str], intent: str,
        confidence: float, slots: dict
        ) -> str | None:
        if len(unique_terms) <= 1 and slots.get("category", {}).get("weight", 0.0) <= self.WARM_THRESHOLD:
            return "category"

        if intent == "buying" and confidence > 0.6 and len(unique_terms) >= 3:
            return None

        # NEW: only ask about a slot if memory doesn't already have a warm
        # value for it — no point re-asking about brand three turns in a row
        for name in ("budget", "brand", "size", "color"):
            weight = slots.get(name, {}).get("weight", 0.0)
            if weight <= self.WARM_THRESHOLD:
                return name

        return None

    def _resolve_message(
        self, intent: str, ask_attribute: str | None, recommendations: list[dict]
    ) -> str:
        if ask_attribute is not None:
            prompts = {
                "category": "What kind of product are you looking for?",
                "budget": "Do you have a budget in mind?",
                "brand": "Any brand preference, or open to anything?",
                "size": "What size are you looking for?",
                "color": "Any color preference?",
            }
            return prompts.get(ask_attribute, "Could you tell me a bit more about what you need?")
        if not recommendations:
            return "I couldn't find a close match — want to tell me more about what you need?"
        if intent == "buying":
            return "Here are the closest matches to buy."
        return "Here are a few options to explore."

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        memory = self._sessions[session_id]
        memory["turn"] = turn

        for slot in memory["slots"].values():
            slot["weight"] *= self.DECAY_RATE

        self._extract_slots(user_message, memory["slots"])

        intent, confidence = _detect_intent(user_message)
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]

        # CHANGED: now self.-methods, and pass memory["slots"] through
        ask_attribute = self._resolve_ask_attribute(
            user_message, unique_terms, intent, confidence, memory["slots"]
        )

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            effective_top_k = self._resolve_top_k(intent, confidence, top_k)
            if ask_attribute is not None:
                effective_top_k = min(effective_top_k, 3)

            weights = self._resolve_weights(intent, confidence, memory["slots"])  

            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weights}) LIMIT ?",
                (expression, effective_top_k * 2),  # over-fetch to survive filtering
            ).fetchall()
            recommendations = [
                {"parent_asin": str(row[0])} for row in rows
                if str(row[0]) not in memory["rejected_asins"]
            ][:effective_top_k]
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
            for r in recommendations:
                memory["rejected_asins"].add(r["parent_asin"])

        message = self._resolve_message(intent, ask_attribute, recommendations)

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }