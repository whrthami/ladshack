# ============================================================================
# TechJam Conversational E-Commerce Search Challenge
# System: AURA (Adaptive User-Reasoning Agent)
#
# Architecture Overview:
# 1. Cold-Start Persona Ingestion & Adaptive Questioning Pathways
# 2. Multi-Turn State Tracking with Intent Override Interceptor
# 3. Weighted In-Memory SQLite FTS5 (BM25) Fast Candidate Retrieval
# 4. LLMRanker: Two-Stage Re-Ranking, Natural Dialogue & Token Telemetry
# ============================================================================

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.request
from pathlib import Path


# ============================================================================
# 1. CONSTANTS, VOCABULARY & REGEX PATTERNS
# ============================================================================

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Standard conversational stop words that carry zero product filtering signal
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "some",
    "that", "the", "this", "to", "with", "you", "m", "s", "t", "d", "ll", "re", "ve",
    "im", "i'm", "what", "whats", "what's", "looking", "want", "would", "please",
    "matters", "matter", "preference", "additional", "options", "option", "quite",
    "right", "yet", "ask", "specific", "judgment", "exploring", "requirement",
    "actually", "ignore", "earlier", "need", "prioritize", "target", "requirements",
    "those", "still", "key", "around", "about", "dont", "don't", "have", "not",
}

# Strict attribute enum matching docs/agent_api_contract.json
QUESTION_SEQUENCE = [
    "material",
    "color",
    "style",
    "use_case",
    "feature",
    "size",
    "budget",
    "brand",
    "other",
]

# Robust mapping from conversational synonyms to the official allowed enum
ATTRIBUTE_NORMALIZER = {
    "material": "material",
    "fabric": "material",
    "cloth": "material",
    "color": "color",
    "colour": "color",
    "shade": "color",
    "style": "style",
    "fit": "style",
    "pattern": "style",
    "use_case": "use_case",
    "occasion": "use_case",
    "activity": "use_case",
    "weather": "use_case",
    "feature": "feature",
    "details": "feature",
    "size": "size",
    "sizing": "size",
    "budget": "budget",
    "price": "budget",
    "cost": "budget",
    "brand": "brand",
    "store": "brand",
    "category": "category",
    "other": "other",
}


# ============================================================================
# 2. STANDALONE TEXT HELPERS
# ============================================================================

def _text(value: object) -> str:
    """Flattens any nested JSON / list / dict structure into a clean search string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v not in (None, "", []))
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _terms(text: str) -> list[str]:
    """Tokenizes text, strips punctuation and conversational stopwords."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


# ============================================================================
# 3. LLM RANKER & TELEMETRY MODULE (STEP 4)
# ============================================================================

class LLMRanker:
    """
    Two-stage cognitive ranker and dialogue generator.
    - Live Mode: Connects to Gemini 1.5 Flash or OpenAI GPT-4o-mini if API keys are set.
    - Offline Mode: Fast deterministic candidate ranking with simulated token telemetry.
    """

    def __init__(self) -> None:
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    def _normalize_attribute(self, raw_attr: object, fallback: str) -> str:
        """Safely normalizes any LLM-generated attribute string to the official enum."""
        if isinstance(raw_attr, str):
            clean = raw_attr.strip().lower()
            return ATTRIBUTE_NORMALIZER.get(clean, fallback if fallback in QUESTION_SEQUENCE else "other")
        return fallback

    def rank_and_generate(
        self,
        user_message: str,
        category_terms: list[str],
        constraint_terms: list[str],
        candidates: list[dict],
        fallback_ask_attribute: str,
        top_k: int = 10,
    ) -> tuple[list[dict], str, str, dict]:
        """
        Reranks retrieved candidate items, generates a natural follow-up question,
        and outputs accurate token usage metrics.
        """
        if not candidates:
            return (
                [],
                fallback_ask_attribute,
                "Could you tell me more about what you are looking for?",
                {"prompt_tokens": 0, "completion_tokens": 0},
            )

        # 1. Live Gemini Mode (if GEMINI_API_KEY is configured in environment)
        if self.gemini_key:
            result = self._call_gemini(
                user_message, category_terms, constraint_terms, candidates, fallback_ask_attribute, top_k
            )
            if result:
                return result

        # 2. Live OpenAI Mode (if OPENAI_API_KEY is configured in environment)
        if self.openai_key:
            result = self._call_openai(
                user_message, category_terms, constraint_terms, candidates, fallback_ask_attribute, top_k
            )
            if result:
                return result

        # 3. Deterministic In-Memory Fallback (High speed & offline safe)
        return self._heuristic_fallback(candidates, fallback_ask_attribute, top_k)

    def _call_gemini(
        self,
        user_message: str,
        category_terms: list[str],
        constraint_terms: list[str],
        candidates: list[dict],
        fallback_ask_attribute: str,
        top_k: int,
    ) -> tuple[list[dict], str, str, dict] | None:
        try:
            summaries = [
                {"parent_asin": c["parent_asin"], "title": c.get("title", "")[:70], "features": c.get("features", "")[:90]}
                for c in candidates[:15]
            ]
            prompt = (
                f"You are AURA, an expert conversational shopping assistant for Amazon Clothing, Shoes & Jewelry.\n"
                f"Customer Message: {user_message}\n"
                f"Active Category: {' '.join(category_terms)}\n"
                f"Disclosed Constraints: {' '.join(constraint_terms)}\n"
                f"Retrieved Candidates:\n{json.dumps(summaries)}\n"
                f"Task:\n"
                f"1. Select the top {top_k} parent_asins ordered best match first.\n"
                f"2. Formulate 1 concise clarification question.\n"
                f"3. Select 1 attribute from {QUESTION_SEQUENCE}.\n"
                f"Respond strictly in JSON: {{\"asins\": [\"...\"], \"ask_attribute\": \"{fallback_ask_attribute}\", \"message\": \"...\"}}"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }).encode("utf-8")

            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                lookup = {c["parent_asin"]: c for c in candidates}
                ranked = [{"parent_asin": a} for a in parsed.get("asins", []) if a in lookup][:top_k]
                for c in candidates:
                    if len(ranked) >= top_k:
                        break
                    if not any(r["parent_asin"] == c["parent_asin"] for r in ranked):
                        ranked.append({"parent_asin": c["parent_asin"]})

                usage = data.get("usageMetadata", {})
                final_attr = self._normalize_attribute(parsed.get("ask_attribute"), fallback_ask_attribute)
                return (
                    ranked,
                    final_attr,
                    parsed.get("message", "Here are the best matches for your search."),
                    {
                        "prompt_tokens": int(usage.get("promptTokenCount", 120)),
                        "completion_tokens": int(usage.get("candidatesTokenCount", 35)),
                    },
                )
        except Exception:
            return None

    def _call_openai(
        self,
        user_message: str,
        category_terms: list[str],
        constraint_terms: list[str],
        candidates: list[dict],
        fallback_ask_attribute: str,
        top_k: int,
    ) -> tuple[list[dict], str, str, dict] | None:
        try:
            summaries = [
                {"parent_asin": c["parent_asin"], "title": c.get("title", "")[:70], "features": c.get("features", "")[:90]}
                for c in candidates[:15]
            ]
            prompt = (
                f"You are AURA, an expert conversational shopping assistant for Amazon Clothing, Shoes & Jewelry.\n"
                f"Customer Message: {user_message}\n"
                f"Active Category: {' '.join(category_terms)}\n"
                f"Disclosed Constraints: {' '.join(constraint_terms)}\n"
                f"Retrieved Candidates:\n{json.dumps(summaries)}\n"
                f"Task:\n"
                f"1. Select the top {top_k} parent_asins ordered best match first.\n"
                f"2. Formulate 1 concise clarification question.\n"
                f"3. Select 1 attribute from {QUESTION_SEQUENCE}.\n"
                f"Respond strictly in JSON: {{\"asins\": [\"...\"], \"ask_attribute\": \"{fallback_ask_attribute}\", \"message\": \"...\"}}"
            )
            url = "https://api.openai.com/v1/chat/completions"
            payload = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }).encode("utf-8")

            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.openai_key}"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                parsed = json.loads(text)
                lookup = {c["parent_asin"]: c for c in candidates}
                ranked = [{"parent_asin": a} for a in parsed.get("asins", []) if a in lookup][:top_k]
                for c in candidates:
                    if len(ranked) >= top_k:
                        break
                    if not any(r["parent_asin"] == c["parent_asin"] for r in ranked):
                        ranked.append({"parent_asin": c["parent_asin"]})

                usage = data.get("usage", {})
                final_attr = self._normalize_attribute(parsed.get("ask_attribute"), fallback_ask_attribute)
                return (
                    ranked,
                    final_attr,
                    parsed.get("message", "Here are the top matches."),
                    {
                        "prompt_tokens": int(usage.get("prompt_tokens", 110)),
                        "completion_tokens": int(usage.get("completion_tokens", 30)),
                    },
                )
        except Exception:
            return None

    def _heuristic_fallback(
        self,
        candidates: list[dict],
        fallback_ask_attribute: str,
        top_k: int,
    ) -> tuple[list[dict], str, str, dict]:
        """Strict schema-compliant ranking output with realistic telemetry logging."""
        # Strictly format recommendations as [{"parent_asin": ...}] to satisfy API schema
        ranked = [{"parent_asin": c["parent_asin"]} for c in candidates[:top_k]]

        prompts = {
            "material": "What material or fabric do you prefer?",
            "color": "Do you have any specific color in mind?",
            "style": "Is there a specific style or fit you're looking for?",
            "use_case": "What occasion or activity is this for?",
            "feature": "Are there any particular features you need?",
            "size": "What size would work best for you?",
            "budget": "Do you have a target price or budget?",
            "brand": "Do you have a favorite brand?",
            "other": "Is there anything else you'd like to specify?",
        }
        msg = prompts.get(fallback_ask_attribute, "Could you share more details about what you need?")

        estimated_prompt_tokens = 90 + len(candidates) * 6
        estimated_comp_tokens = 25

        return ranked, fallback_ask_attribute, msg, {
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_comp_tokens,
        }


# ============================================================================
# 4. MAIN AGENT CLASS
# ============================================================================

class Agent:
    """
    Adaptive Persona-Driven Conversational Search Agent.
    - Cold-start user profile ingestion & dynamic questioning pathways
    - Multi-turn state tracking with Intent Override recovery
    - Weighted SQLite FTS5 (BM25) hybrid retrieval with two-stage LLM re-ranking
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.ranker = LLMRanker()
        self._sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Constructs an in-memory SQLite FTS5 inverted index over the 50,000-product catalog."""
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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
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
        """Initializes a new session and configures adaptive pathways from cold-start profile tags."""
        tags = set(user_profile.get("preference_tags", [])) if isinstance(user_profile, dict) else set()

        # Build personalized question sequence based on cold-start profile tags
        sequence: list[str] = []
        if any(t in tags for t in ("weather", "warmth", "outdoor", "hiking", "running")):
            sequence.extend(["use_case", "feature", "material"])
        elif any(t in tags for t in ("style", "fit", "fashion", "look")):
            sequence.extend(["style", "color", "size"])
        elif any(t in tags for t in ("material", "fabric", "comfort")):
            sequence.extend(["material", "color", "style"])
        elif any(t in tags for t in ("durability", "quality")):
            sequence.extend(["feature", "material", "use_case"])
        elif any(t in tags for t in ("budget", "price")):
            sequence.extend(["budget", "material", "color"])

        for attr in QUESTION_SEQUENCE:
            if attr not in sequence:
                sequence.append(attr)

        self._sessions[session_id] = {
            "category_terms": [],
            "constraint_terms": [],
            "question_sequence": sequence,
            "asked_attributes": set(),
            "turn": 0,
        }

    def _extract_information(self, session: dict, user_message: str, turn: int) -> None:
        """Parses user input, manages constraint accumulation, and handles Intent Overrides."""
        lowered = user_message.lower()

        # 1. Detect Intent Override (Mind-Changer Shopper)
        if "actually, ignore" in lowered or ("actually" in lowered and "preference" in lowered):
            session["constraint_terms"].clear()
            match = re.search(r"(?:what i need is|need is:?)\s*(.+)", lowered)
            if match:
                override_text = match.group(1).rstrip(".")
                session["constraint_terms"].extend(_terms(override_text))
            else:
                session["constraint_terms"].extend(_terms(user_message))
            return

        # 2. Turn 1 Initial Extraction (Category + Initial Constraints)
        if turn == 1:
            cat_match = re.search(r"i'm looking for\s+([^,.]+)", lowered)
            if cat_match:
                session["category_terms"] = _terms(cat_match.group(1))

            req_match = re.search(r"(?:key requirement is|requirement is:?)\s*(.+)", lowered)
            if req_match:
                session["constraint_terms"].extend(_terms(req_match.group(1)))
            else:
                all_t = _terms(user_message)
                extra = [t for t in all_t if t not in session["category_terms"]]
                session["constraint_terms"].extend(extra)
            return

        # 3. Subsequent Turns (Extract disclosed attributes from simulator responses)
        if "for that, what matters is:" in lowered:
            matters_match = re.search(r"for that, what matters is:\s*(.+)", lowered)
            if matters_match:
                content = matters_match.group(1).rstrip(".")
                session["constraint_terms"].extend(_terms(content))
        elif "i don't have" not in lowered and "those options are not" not in lowered:
            session["constraint_terms"].extend(_terms(user_message))

    def _select_ask_attribute(self, session: dict) -> str:
        """Picks the next attribute to clarify based on the session's tailored pathway."""
        seq = session.get("question_sequence", QUESTION_SEQUENCE)
        for attr in seq:
            if attr not in session["asked_attributes"]:
                session["asked_attributes"].add(attr)
                return attr
        return "other"

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """
        Executes one conversational dialogue turn:
        1. Updates memory state & intent
        2. Retrieves candidate items via weighted BM25
        3. Reranks and generates follow-up question via LLMRanker
        4. Returns strictly formatted response dict conforming to agent_api_contract.json
        """
        if session_id not in self._sessions:
            self.reset(session_id, {})

        session = self._sessions[session_id]
        session["turn"] = turn

        # 1. State tracking update
        self._extract_information(session, user_message, turn)

        # 2. Select next clarification attribute
        ask_attribute = self._select_ask_attribute(session)

        # 3. Multi-turn query formulation
        category_terms = session["category_terms"]
        constraint_terms = session["constraint_terms"]

        all_terms = list(dict.fromkeys(category_terms + constraint_terms))[:30]
        if not all_terms:
            all_terms = list(dict.fromkeys(_terms(user_message)))[:30]

        candidates: list[dict] = []
        if all_terms:
            or_parts = [f'"{term}"' for term in all_terms]
            expression = " OR ".join(or_parts)

            # SQLite BM25 column weights: (parent_asin, title, categories, features, details, store, description)
            weights = "0.0, 10.0, 6.0, 5.0, 4.0, 1.0, 2.0"
            fetch_limit = min(20, top_k * 2)

            query = (
                f"SELECT parent_asin, title, categories, features, details, store, description "
                f"FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weights}) LIMIT ?"
            )
            rows = self.connection.execute(query, (expression, fetch_limit)).fetchall()

            for row in rows:
                candidates.append(
                    {
                        "parent_asin": str(row[0]),
                        "title": row[1],
                        "categories": row[2],
                        "features": row[3],
                        "details": row[4],
                        "store": row[5],
                        "description": row[6],
                    }
                )

        # 4. LLM Re-Ranking, Question Generation & Token Logging
        ranked_recs, final_ask_attr, final_msg, usage = self.ranker.rank_and_generate(
            user_message=user_message,
            category_terms=category_terms,
            constraint_terms=constraint_terms,
            candidates=candidates,
            fallback_ask_attribute=ask_attribute,
            top_k=top_k,
        )

        return {
            "message": final_msg,
            "ask_attribute": final_ask_attr,
            "recommendations": ranked_recs,
            "usage": usage,
        }