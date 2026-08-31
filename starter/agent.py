from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


# --- Constants & Regex Patterns ---------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "some",
    "that", "the", "this", "to", "with", "you", "im", "what", "matter",
    "matters", "need", "requirement", "preference", "looking", "want",
}

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")

ATTRIBUTE_PRIORITY = ["material", "color", "style", "use_case", "feature", "other"]

OVERRIDE_RE = re.compile(
    r"actually,?\s+ignore\s+(?:my\s+)?earlier\s+preference[.:]?\s*(?:what\s+i\s+need\s+is:\s*)?(.*)",
    re.IGNORECASE,
)

REQUIREMENT_RE = re.compile(
    r"a key requirement is:\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

MATTERS_RE = re.compile(
    r"for that,\s*what matters is:\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

NEED_RE = re.compile(
    r"what i need is:\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

CATEGORY_RE = re.compile(
    r"i'm looking for\s+(.+?)(?:\. A key requirement|\. |, but I'm still exploring|\.|$)",
    re.IGNORECASE,
)

REJECTION_PATTERNS = re.compile(
    r"\bnot that\b|\bsomething else\b|\bdon'?t like\b|\bskip\b|\bno thanks\b",
    re.IGNORECASE,
)

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


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items() if item is not None)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _detect_intent(text: str) -> tuple[str, float]:
    lowered = text.lower()
    raw_tokens = set(TOKEN_RE.findall(lowered))
    buy_score = len(raw_tokens & BUYING_WORDS)
    browse_score = len(raw_tokens & BROWSING_WORDS)
    if "key requirement" in lowered:
        buy_score += 3
    if "exploring" in lowered:
        browse_score += 3

    total = buy_score + browse_score
    if total == 0:
        return "browsing", 0.5
    if buy_score >= browse_score:
        return "buying", buy_score / total
    return "browsing", browse_score / total


# --- Agent Implementation ---------------------------------------------------

class Agent:
    """Multi-turn e-commerce retrieval agent with FTS5 candidate generation and lexical/constraint reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._sessions: dict[str, dict] = {}
        self._products: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                avg_rating = float(product.get("average_rating") or 0.0)
                rating_num = int(product.get("rating_number") or 0)

                self._products[parent_asin] = {
                    "parent_asin": parent_asin,
                    "title": title,
                    "categories": categories,
                    "features": features,
                    "details": details,
                    "store": store,
                    "description": description,
                    "average_rating": avg_rating,
                    "rating_number": rating_num,
                }

                batch.append((parent_asin, title, categories, features, details, store, description))
                if len(batch) >= 2000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()

        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "user_profile": user_profile or {},
            "category": "",
            "category_terms": [],
            "constraints": [],
            "asked_attributes": set(),
            "rejected_asins": set(),
            "last_shown": [],
            "turn": 0,
            "override_happened": False,
        }

    def _extract_information(self, user_message: str, memory: dict) -> None:
        # 1. Intent Override check
        override_match = OVERRIDE_RE.search(user_message)
        if override_match:
            memory["override_happened"] = True
            new_val = override_match.group(1).strip(" .;")
            memory["constraints"] = [new_val] if new_val else []
            return

        # 2. Extract Category on initial turns
        if not memory["category"]:
            cat_match = CATEGORY_RE.search(user_message)
            if cat_match:
                cat_raw = cat_match.group(1).strip(" .;")
                # Clean filler like "some" or "a"
                cat_clean = re.sub(r"^(?:a|an|some)\s+", "", cat_raw, flags=re.I).strip()
                memory["category"] = cat_clean
                memory["category_terms"] = _terms(cat_clean)

        # 3. Extract explicit requirements / constraints
        req_match = REQUIREMENT_RE.search(user_message)
        if req_match:
            req = req_match.group(1).strip(" .;")
            if req and req not in memory["constraints"]:
                memory["constraints"].append(req)

        matters_match = MATTERS_RE.search(user_message)
        if matters_match:
            raw_items = matters_match.group(1).split(";")
            for item in raw_items:
                clean_item = item.strip(" .;")
                if clean_item and clean_item not in memory["constraints"]:
                    memory["constraints"].append(clean_item)

        need_match = NEED_RE.search(user_message)
        if need_match:
            need = need_match.group(1).strip(" .;")
            if need and need not in memory["constraints"]:
                memory["constraints"].append(need)

    def _select_ask_attribute(self, memory: dict) -> str | None:
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in memory["asked_attributes"]:
                memory["asked_attributes"].add(attr)
                return attr
        return "other"

    def _score_candidate(
        self,
        candidate_asin: str,
        bm25_score: float,
        memory: dict,
    ) -> float:
        prod = self._products.get(candidate_asin)
        if not prod:
            return -9999.0

        title = prod["title"]
        features = prod["features"]
        details = prod["details"]
        categories = prod["categories"]
        desc = prod["description"]

        title_lower = title.lower()
        features_lower = features.lower()
        details_lower = details.lower()
        categories_lower = categories.lower()
        desc_lower = desc.lower()
        corpus = f"{title_lower} {features_lower} {details_lower} {categories_lower} {desc_lower}"

        # SQLite FTS5 BM25 score is negative (lower = better), so -bm25_score is positive
        score = max(0.0, -bm25_score) * 1.0

        # Category alignment
        category_terms = memory["category_terms"]
        if category_terms:
            title_hits = sum(1 for t in category_terms if t in title_lower)
            cat_hits = sum(1 for t in category_terms if t in categories_lower)

            if title_hits == len(category_terms):
                score += 15.0
            elif title_hits > 0:
                score += (title_hits / len(category_terms)) * 10.0
            score += cat_hits * 2.0

        # Disclosed Constraints matching
        for constraint in memory["constraints"]:
            c_lower = constraint.lower()
            # If constraint mentions color: remove prefix "color: " for exact check
            c_clean = re.sub(r"^color:\s*", "", c_lower).strip()

            if c_clean in features_lower or c_clean in details_lower:
                score += 30.0
            elif c_clean in title_lower:
                score += 25.0
            elif c_clean in desc_lower:
                score += 12.0
            else:
                c_tokens = [t for t in TOKEN_RE.findall(c_clean) if t not in STOPWORDS and len(t) > 1]
                if c_tokens:
                    matched_tokens = sum(1 for t in c_tokens if t in corpus)
                    score += (matched_tokens / len(c_tokens)) * 15.0
                    feat_matched = sum(1 for t in c_tokens if t in features_lower or t in title_lower)
                    score += (feat_matched / len(c_tokens)) * 10.0

        # User profile alignment (helpful in browsing sessions)
        user_profile = memory.get("user_profile", {})
        for tag in user_profile.get("preference_tags", []):
            t_lower = str(tag).lower()
            if t_lower in features_lower or t_lower in details_lower or t_lower in title_lower:
                score += 1.5

        # Rating prior
        avg_rating = prod["average_rating"]
        num_ratings = min(prod["rating_number"], 1000)
        score += (avg_rating / 5.0) * 1.5 + (num_ratings / 1000.0) * 1.0

        return score

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

        # Check for user rejection
        if REJECTION_PATTERNS.search(user_message):
            memory["rejected_asins"].update(memory["last_shown"])

        # Extract information from current message
        self._extract_information(user_message, memory)

        intent, confidence = _detect_intent(user_message)
        ask_attribute = self._select_ask_attribute(memory)

        # Build search query terms
        query_terms: list[str] = []
        if memory["category_terms"]:
            query_terms.extend(memory["category_terms"])

        for c in memory["constraints"]:
            query_terms.extend(_terms(c))

        # Fallback if no specific terms yet
        if not query_terms:
            query_terms = _terms(user_message)

        query_terms = list(dict.fromkeys(query_terms))[:30]

        if not query_terms:
            recommendations: list[dict] = []
        else:
            fts_expression = " OR ".join(f'"{term}"' for term in query_terms)
            query = (
                "SELECT parent_asin, bm25(products, 6.0, 4.0, 4.0, 3.0, 1.5, 1.0) "
                "FROM products "
                "WHERE products MATCH ? "
                "ORDER BY bm25(products, 6.0, 4.0, 4.0, 3.0, 1.5, 1.0) "
                "LIMIT 100"
            )
            candidates = self.connection.execute(query, (fts_expression,)).fetchall()

            scored_candidates: list[tuple[float, str]] = []
            for asin, bm25_score in candidates:
                if asin in memory["rejected_asins"]:
                    continue
                score = self._score_candidate(asin, bm25_score, memory)
                scored_candidates.append((score, asin))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            recommendations = []
            for _, asin in scored_candidates[:top_k]:
                prod = self._products[asin]
                recommendations.append({
                    "parent_asin": asin,
                    "title": prod["title"],
                    "categories": prod["categories"],
                    "features": prod["features"],
                    "details": prod["details"],
                    "store": prod["store"],
                    "description": prod["description"],
                })

        memory["last_shown"] = [r["parent_asin"] for r in recommendations]

        # Construct clarification / response message
        prompts = {
            "material": "Do you have a specific material in mind?",
            "color": "Is there a preferred color you're looking for?",
            "style": "Any specific style or cut preference?",
            "use_case": "What occasion or activity is this for?",
            "feature": "Are there any special features or details you require?",
            "other": "Are there any other specific requirements you have?",
        }
        lead_msg = prompts.get(ask_attribute, "Could you tell me more about what you're looking for?")
        if recommendations:
            lead_msg += " Here are my top recommendations."

        return {
            "message": lead_msg,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }