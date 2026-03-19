"""Thread/message storage for short-term and long-term conversation persistence."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class ConversationStore:
    def __init__(self, path: str = "data/cache/conversations.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"threads": {}})

    def create_thread(self, title: str) -> str:
        data = self._load()
        thread_id = datetime.utcnow().strftime("thread_%Y%m%d_%H%M%S")
        while thread_id in data["threads"]:
            time.sleep(0.01)
            thread_id = datetime.utcnow().strftime("thread_%Y%m%d_%H%M%S")

        now = self._now()
        data["threads"][thread_id] = {
            "thread_id": thread_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "context": {},
            "messages": [],
        }
        self._save(data)
        return thread_id

    def list_threads(self) -> list[dict[str, Any]]:
        data = self._load()
        threads = list(data["threads"].values())
        threads.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return threads

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        data = self._load()
        thread = data["threads"].get(thread_id)
        if not thread:
            raise KeyError(f"Unknown thread_id: {thread_id}")
        return thread

    def add_message(self, thread_id: str, role: str, content: str, summary: str = ""):
        data = self._load()
        thread = data["threads"].get(thread_id)
        if not thread:
            raise KeyError(f"Unknown thread_id: {thread_id}")

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "ts": self._now(),
            "role": role,
            "content": content,
            "summary": summary,
        }
        thread["messages"].append(msg)
        thread["updated_at"] = self._now()
        self._save(data)

    def update_context(self, thread_id: str, updates: dict[str, Any]):
        data = self._load()
        thread = data["threads"].get(thread_id)
        if not thread:
            raise KeyError(f"Unknown thread_id: {thread_id}")
        ctx = thread.setdefault("context", {})
        ctx.update(updates)
        thread["updated_at"] = self._now()
        self._save(data)

    def get_short_term(self, thread_id: str, n: int = 6) -> list[dict[str, Any]]:
        thread = self.get_thread(thread_id)
        return thread.get("messages", [])[-n:]

    def get_long_term(self, thread_id: str) -> list[dict[str, Any]]:
        thread = self.get_thread(thread_id)
        return thread.get("messages", [])

    def delete_thread(self, thread_id: str) -> bool:
        data = self._load()
        if thread_id not in data["threads"]:
            return False
        del data["threads"][thread_id]
        self._save(data)
        return True

    def delete_all_except(self, keep_thread_id: str) -> int:
        data = self._load()
        keys = list(data["threads"].keys())
        removed = 0
        for tid in keys:
            if tid == keep_thread_id:
                continue
            del data["threads"][tid]
            removed += 1
        self._save(data)
        return removed

    def _load(self) -> dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, payload: dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
