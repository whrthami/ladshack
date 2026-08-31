from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import uuid
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from starter.agent import Agent
from evaluator.local_evaluator import (
    load_jsonl,
    catalog_index,
    coarse_category,
    materialize_hidden_fields,
    initial_message,
    customer_reply,
    normalize_recommendations,
)


class ShoppingAgentServer:
    def __init__(self, catalog_path: str = "data/catalog.jsonl", dataset_path: str = "data/public_set.jsonl"):
        print(f"[*] Initializing Agent and Catalog from {catalog_path}...")
        self.agent = Agent(catalog_path)
        self.catalog_ids, self.categories, self.products = catalog_index(catalog_path)
        self.dataset_path = Path(dataset_path)
        self.samples: list[dict] = []
        self.samples_by_id: dict[str, dict] = {}
        if self.dataset_path.exists():
            for s in load_jsonl(self.dataset_path):
                self.samples.append(s)
                self.samples_by_id[s["sample_id"]] = s
        print(f"[*] Loaded {len(self.samples)} sample sessions.")
        self.active_scenarios: dict[str, dict] = {}


class RequestHandler(BaseHTTPRequestHandler):
    server: ShoppingAgentHTTPServer

    def log_message(self, format, *args):
        # Clean formatted logging in terminal
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")
        sys.stdout.flush()

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("", "/", "/index.html"):
            html_path = Path("ui/index.html")
            if html_path.exists():
                self._set_headers(200, "text/html; charset=utf-8")
                self.wfile.write(html_path.read_bytes())
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"ui/index.html not found")
            return

        if path == "/api/samples":
            samples_summary = []
            for s in self.server.app.samples:
                target_asin = s.get("ground_truth", {}).get("parent_asin")
                target_prod = self.server.app.products.get(target_asin, {})
                samples_summary.append({
                    "sample_id": s.get("sample_id"),
                    "scenario_type": s.get("scenario_type"),
                    "difficulty_bucket": s.get("difficulty_bucket"),
                    "category_bucket": s.get("category_bucket"),
                    "target_asin": target_asin,
                    "target_title": target_prod.get("title", "Product"),
                    "user_profile": s.get("user_profile", {}),
                })
            self._set_headers(200)
            self.wfile.write(json.dumps({"samples": samples_summary}).encode("utf-8"))
            return

        static_file = Path("ui" + path)
        if static_file.exists() and static_file.is_file():
            mime_type, _ = mimetypes.guess_type(str(static_file))
            self._set_headers(200, mime_type or "application/octet-stream")
            self.wfile.write(static_file.read_bytes())
            return

        self._set_headers(404, "text/plain")
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/reset":
            session_id = payload.get("session_id", f"sess_{uuid.uuid4().hex[:8]}")
            user_profile = payload.get("user_profile", {})
            self.server.app.agent.reset(session_id, user_profile)
            if session_id in self.server.app.active_scenarios:
                del self.server.app.active_scenarios[session_id]
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "session_id": session_id}).encode("utf-8"))
            return

        if path == "/api/chat":
            session_id = payload.get("session_id", "default_session")
            user_message = payload.get("message", "")
            turn = int(payload.get("turn", 1))
            top_k = int(payload.get("top_k", 10))

            if session_id not in self.server.app.agent._sessions:
                self.server.app.agent.reset(session_id, {})

            try:
                response = self.server.app.agent.respond(session_id, user_message, turn, top_k)
                memory = self.server.app.agent._sessions.get(session_id, {})
                
                enriched_recs = []
                for item in response.get("recommendations", []):
                    asin = item.get("parent_asin")
                    prod = self.server.app.products.get(asin, {}) or self.server.app.agent._products.get(asin, {})
                    enriched_recs.append({
                        "parent_asin": asin,
                        "title": prod.get("title") or item.get("title"),
                        "categories": prod.get("categories") or item.get("categories"),
                        "features": prod.get("features") or item.get("features"),
                        "details": prod.get("details") or item.get("details"),
                        "store": prod.get("store") or item.get("store"),
                        "description": prod.get("description") or item.get("description"),
                        "average_rating": prod.get("average_rating", 0.0),
                        "rating_number": prod.get("rating_number", 0),
                        "price": prod.get("price"),
                    })

                debug_state = {
                    "category": memory.get("category"),
                    "constraints": memory.get("constraints", []),
                    "asked_attributes": list(memory.get("asked_attributes", [])),
                    "turn": memory.get("turn", turn),
                }

                result = {
                    "message": response.get("message"),
                    "ask_attribute": response.get("ask_attribute"),
                    "recommendations": enriched_recs,
                    "debug": debug_state,
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # Scenario Start: Initialize a simulated test case
        if path == "/api/scenario/start":
            sample_id = payload.get("sample_id")
            sample = self.server.app.samples_by_id.get(sample_id)
            if not sample:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Sample {sample_id} not found"}).encode("utf-8"))
                return

            session_id = f"sim_{sample_id}_{uuid.uuid4().hex[:6]}"
            self.server.app.agent.reset(session_id, sample.get("user_profile", {}))
            
            target_asin = str(sample["ground_truth"]["parent_asin"])
            target_prod = self.server.app.products.get(target_asin, {})
            card, behavior = materialize_hidden_fields(sample, self.server.app.products)
            effective_sample = {**sample, "intent_card": card, "behavior": behavior}

            disclosed = set()
            coarse_cat = coarse_category(self.server.app.categories.get(target_asin, []))
            first_user_msg = initial_message(effective_sample, coarse_cat, disclosed)

            self.server.app.active_scenarios[session_id] = {
                "effective_sample": effective_sample,
                "target_asin": target_asin,
                "disclosed": disclosed,
                "boundary_used": False,
                "override_applied": sample["scenario_type"] != "intent_override",
                "next_user_message": first_user_msg,
                "turn": 1,
                "is_hit": False,
                "hit_turn": None,
                "hit_rank": None,
            }

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "session_id": session_id,
                "sample_id": sample_id,
                "scenario_type": sample["scenario_type"],
                "target_asin": target_asin,
                "target_title": target_prod.get("title", ""),
                "target_categories": self.server.app.categories.get(target_asin, []),
                "hard_constraints": card.get("hard_constraints", []),
                "soft_preferences": card.get("soft_preferences", []),
                "initial_customer_message": first_user_msg,
            }).encode("utf-8"))
            return

        # Scenario Step: Advance simulated scenario by 1 turn
        if path == "/api/scenario/step":
            session_id = payload.get("session_id")
            state = self.server.app.active_scenarios.get(session_id)
            if not state:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No active scenario session"}).encode("utf-8"))
                return

            if state["is_hit"] or state["turn"] > 10:
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "finished": True,
                    "is_hit": state["is_hit"],
                    "hit_turn": state["hit_turn"],
                    "hit_rank": state["hit_rank"],
                }).encode("utf-8"))
                return

            turn = state["turn"]
            user_msg = state["next_user_message"]
            effective_sample = state["effective_sample"]
            target_asin = state["target_asin"]

            # Agent responds
            response = self.server.app.agent.respond(session_id, user_msg, turn, 10)
            ranked = normalize_recommendations(response.get("recommendations"), self.server.app.catalog_ids)

            # Check hit
            is_hit = False
            best_rank = None
            if state["override_applied"] and target_asin in ranked:
                is_hit = True
                best_rank = ranked.index(target_asin) + 1
                state["is_hit"] = True
                state["hit_turn"] = turn
                state["hit_rank"] = best_rank

            # Prepare next turn customer reply
            next_user_msg = ""
            if not is_hit and turn < 10:
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not state["override_applied"] and turn + 1 == int(override.get("turn", 3)):
                    state["override_applied"] = True
                    new_val = str(override.get("new_value", ""))
                    if new_val:
                        state["disclosed"].add(new_val)
                    next_user_msg = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    next_user_msg, state["boundary_used"] = customer_reply(
                        effective_sample, response.get("ask_attribute"), state["disclosed"], state["boundary_used"]
                    )
                state["next_user_message"] = next_user_msg

            state["turn"] += 1

            enriched_recs = []
            for item in response.get("recommendations", []):
                asin = item.get("parent_asin")
                prod = self.server.app.products.get(asin, {}) or self.server.app.agent._products.get(asin, {})
                enriched_recs.append({
                    "parent_asin": asin,
                    "title": prod.get("title") or item.get("title"),
                    "categories": prod.get("categories") or item.get("categories"),
                    "features": prod.get("features") or item.get("features"),
                    "details": prod.get("details") or item.get("details"),
                    "store": prod.get("store") or item.get("store"),
                    "description": prod.get("description") or item.get("description"),
                    "average_rating": prod.get("average_rating", 0.0),
                    "rating_number": prod.get("rating_number", 0),
                    "price": prod.get("price"),
                    "is_target": asin == target_asin,
                })

            memory = self.server.app.agent._sessions.get(session_id, {})
            result = {
                "turn": turn,
                "customer_message": user_msg,
                "agent_message": response.get("message"),
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": enriched_recs,
                "is_hit": is_hit,
                "hit_rank": best_rank,
                "next_customer_message": next_user_msg,
                "finished": is_hit or turn >= 10,
                "debug": {
                    "category": memory.get("category"),
                    "constraints": memory.get("constraints", []),
                    "asked_attributes": list(memory.get("asked_attributes", [])),
                    "disclosed": list(state["disclosed"]),
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        self._set_headers(404, "text/plain")
        self.wfile.write(b"Endpoint not found")


class ShoppingAgentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, app: ShoppingAgentServer):
        super().__init__(server_address, RequestHandlerClass)
        self.app = app


def start_server(host: str = "0.0.0.0", port: int = 8000, catalog: str = "data/catalog.jsonl", dataset: str = "data/public_set.jsonl"):
    app = ShoppingAgentServer(catalog_path=catalog, dataset_path=dataset)
    
    current_port = port
    max_attempts = 10
    httpd = None

    for attempt in range(max_attempts):
        try:
            server_address = (host, current_port)
            httpd = ShoppingAgentHTTPServer(server_address, RequestHandler, app)
            break
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 48:
                print(f"[!] Port {current_port} is busy, trying port {current_port + 1}...")
                current_port += 1
            else:
                raise e

    if not httpd:
        print(f"[X] Could not bind to any port from {port} to {current_port}.")
        sys.exit(1)

    print("=" * 60)
    print("🚀 Shopping Agent Frontend is live!")
    print(f"👉 Local Access:   http://localhost:{current_port}")
    print(f"👉 Network Access: http://127.0.0.1:{current_port}")
    print("=" * 60)
    print("[*] Press Ctrl+C to stop the server.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Shopping Agent Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--catalog", default="data/catalog.jsonl", help="Path to catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl", help="Path to public_set.jsonl")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, catalog=args.catalog, dataset=args.dataset)


if __name__ == "__main__":
    main()
