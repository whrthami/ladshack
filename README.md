# Multi-Turn E-Commerce Retrieval & Recommendation Agent

A lightweight, high-performance multi-turn e-commerce recommendation and retrieval agent built in Python. The agent leverages **SQLite FTS5** for fast full-text candidate retrieval and applies custom heuristic constraint tracking, intent detection, and multi-factor re-ranking to deliver highly relevant product recommendations over multi-turn conversational interactions.

---

## Key Features

* **Fast SQLite FTS5 Indexing**: In-memory SQLite database utilizing FTS5 full-text search with Porter stemming, unicode tokenization, and field-weighted BM25 scoring.
* **Conversational Context Tracking**: Maintains per-session memory across turns, tracking categories, explicit user constraints, negative rejections, and past recommendations.
* **Constraint & Requirement Extraction**: Automatically parses explicit user constraints (e.g., material, color, specific feature requirements, and complete preference overrides) using targeted pattern matching.
* **Intent Detection**: Classifies user queries into `browsing` or `buying` intents using lexical signals and confidence scoring.
* **Smart Reranking Pipeline**: Reranks top FTS candidate products by combining:
  * Weighted BM25 catalog match scores.
  * Category alignment heuristics.
  * Disclosed constraint satisfaction across product fields (title, features, details, description).
  * User profile tag preferences.
  * Product rating priors (average rating & review counts).
* **Interactive Clarification System**: Dynamically selects prioritized attributes (`material`, `color`, `style`, `use_case`, `feature`, `other`) to ask follow-up questions and narrow down candidate sets.
* **Zero Heavy Dependencies**: Pure Python standard library implementation (`sqlite3`, `json`, `re`, `pathlib`).

---

## Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Message Input                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 1. Context & Constraint Parser│
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   2. Dynamic Intent & Query   │
                     │          Builder              │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │  3. SQLite FTS5 Candidate     │
                     │      Retrieval (BM25)         │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │  4. Reranking & Constraint    │
                     │         Scoring Engine        │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Top-K Recommendations & Follow-up Prompt                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

The agent expects catalog data in **JSON Lines (`.jsonl`)** format at `data/catalog.jsonl` (or a custom path). Each line represents a JSON object with the following schema:

```json
{
  "parent_asin": "B08N5WRWNW",
  "title": "Men's Cotton Casual Crewneck T-Shirt",
  "categories": "Clothing, Shoes & Jewelry > Men > Shirts",
  "features": "100% Breathable Cotton, Pre-shrunk fabric, Ribbed collar",
  "details": "Material: Cotton, Fit: Regular",
  "store": "Apparel Brand",
  "description": "A classic daily wear t-shirt designed for maximum comfort.",
  "average_rating": 4.5,
  "rating_number": 320
}
```

---

## Installation & Requirements

* **Python Version**: Python 3.8+ (Supports Python 3.10+ type annotations).
* **Dependencies**: Python Standard Library only (`sqlite3` built with FTS5 support, standard in modern Python distributions).

No external PyPI packages are required.

---

## Getting Started

### 1. Catalog Setup

Place your product catalog file formatted as `.jsonl` inside the `data/` directory (e.g., `data/catalog.jsonl`).

### 2. Quick Usage Example

```python
from agent import Agent

# Initialize the agent and load/index the catalog
agent = Agent(catalog_path="data/catalog.jsonl")

# Initialize session state for a user session
session_id = "user_session_001"
user_profile = {
    "preference_tags": ["cotton", "breathable", "casual"]
}

agent.reset(session_id=session_id, user_profile=user_profile)

# Turn 1: Initial broad query
response_1 = agent.respond(
    session_id=session_id,
    user_message="I'm looking for a jacket. A key requirement is: waterproof.",
    turn=1,
    top_k=3
)

print(f"Agent Prompt: {response_1['message']}")
print("Recommendations:")
for item in response_1["recommendations"]:
    print(f" - {item['title']} (ASIN: {item['parent_asin']})")

# Turn 2: User adds specific requirement
response_2 = agent.respond(
    session_id=session_id,
    user_message="What I need is: black.",
    turn=2,
    top_k=3
)

print(f"\nAgent Prompt: {response_2['message']}")
print("Updated Recommendations:")
for item in response_2["recommendations"]:
    print(f" - {item['title']} (ASIN: {item['parent_asin']})")
```

---

## API Reference

### `Agent(catalog_path: str | Path = "data/catalog.jsonl")`
Instantiates the agent, initializes the in-memory SQLite database, creates the FTS5 table, and indexes products from the JSONL catalog.

### `Agent.reset(session_id: str, user_profile: dict) -> None`
Resets memory and state for a specific session ID, including constraints, rejected ASINs, category terms, and asked attributes.

### `Agent.respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict`
Processes a turn of user interaction.

#### Parameters:
* **`session_id`** (`str`): Unique identifier for the active session.
* **`user_message`** (`str`): The raw text message sent by the user.
* **`turn`** (`int`): Current conversation turn number.
* **`top_k`** (`int`): Number of recommended items to return.

#### Returns (`dict`):
```json
{
  "message": "Do you have a specific material in mind? Here are my top recommendations.",
  "ask_attribute": "material",
  "recommendations": [
    {
      "parent_asin": "B08N5WRWNW",
      "title": "...",
      "categories": "...",
      "features": "...",
      "details": "...",
      "store": "...",
      "description": "..."
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0
  }
}
```

---

## Pattern Matching & Extraction Rules

The agent parses explicit natural language phrases to refine retrieval:

* **Category Identification**: Matches `i'm looking for <category>`.
* **Explicit Requirements**: Matches `a key requirement is: <requirement>`.
* **Details & Specifics**: Matches `for that, what matters is: <item1>; <item2>` or `what i need is: <need>`.
* **Preference Overrides**: Detects phrases like `actually, ignore my earlier preference` to purge prior constraints.
* **Negative Feedback**: Detects rejection triggers like `not that`, `don't like`, or `skip`, adding previously shown ASINs to the session rejection set.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
