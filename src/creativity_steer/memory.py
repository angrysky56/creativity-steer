"""Memory storage and retrieval for the grounding system."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    id: str
    created: float
    last_used: float
    uses: int
    kind: str  # "lesson" | "correction" | "options" | "fact"
    content: str  # atomic, final form — NO reasoning
    context: str  # when/where it applies
    tags: list[str]
    impact: float
    alternatives: list[str] = field(default_factory=list)  # MOP: the option space
    embedding: list[float] | None = None
    status: str = "active"  # "active" | "dormant"
    source: str = ""  # turn ref


class MemoryStore(Protocol):
    def write(self, item: MemoryItem) -> None: ...
    def retrieve(
        self, query: str, k: int, include_dormant: bool = False
    ) -> list[MemoryItem]: ...
    def touch(self, ids: list[str]) -> None: ...  # bump last_used/uses
    def decay(
        self, max_idle_seconds: float
    ) -> int: ...  # active -> dormant, never delete
    def all(self) -> list[MemoryItem]: ...


class ChromaMemoryStore:
    """A memory store backed by ChromaDB."""

    def __init__(self, embed_backend, path: str = "results/memory_db"):
        if chromadb is None:
            raise ImportError(
                "chromadb is required for ChromaMemoryStore. Run: uv add chromadb"
            )

        self.embed_backend = embed_backend
        self.path = path

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="creativity_steer_memory", metadata={"hnsw:space": "cosine"}
        )

        # Keep track of active ids for decay/touch
        self._active_ids = set()
        self._load_active_ids()

    def _load_active_ids(self):
        # We need to maintain state of active vs dormant, but Chroma doesn't have an easy "get all"
        # Since we're small-scale, we can fetch all metadata to find active ones
        try:
            results = self.collection.get(include=["metadatas"])
            if results and results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"]):
                    if metadata and metadata.get("status") == "active":
                        self._active_ids.add(results["ids"][i])
        except Exception as e:
            logger.error(f"Failed to load active ids from Chroma: {e}")

    def write(self, item: MemoryItem) -> None:
        """Write an item to memory. Deduplicates near-matches if needed."""
        if not item.embedding:
            # Need embedding for storage
            item.embedding = self.embed_backend.embed([item.content])[0]

        # Convert list attributes to strings for Chroma metadata
        metadata = {
            "created": item.created,
            "last_used": item.last_used,
            "uses": item.uses,
            "kind": item.kind,
            "context": item.context,
            "tags": ",".join(item.tags),
            "impact": item.impact,
            "alternatives": json.dumps(item.alternatives),
            "status": item.status,
            "source": item.source,
        }

        self.collection.add(
            ids=[item.id],
            embeddings=[item.embedding],
            documents=[item.content],
            metadatas=[metadata],
        )

        if item.status == "active":
            self._active_ids.add(item.id)

    def retrieve(
        self, query: str, k: int, include_dormant: bool = False
    ) -> list[MemoryItem]:
        """Retrieve the k most relevant memories."""
        query_embedding = self.embed_backend.embed([query])[0]

        # Filter by status if not including dormant
        where = {} if include_dormant else {"status": "active"}

        # Guard against k > count
        count = self.collection.count()
        if count == 0:
            return []

        k_safe = min(k, count)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k_safe,
                where=where,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]

            # Parse complex metadata back
            tags = meta["tags"].split(",") if meta["tags"] else []
            alternatives = (
                json.loads(meta["alternatives"]) if "alternatives" in meta else []
            )

            item = MemoryItem(
                id=doc_id,
                created=meta["created"],
                last_used=meta["last_used"],
                uses=meta["uses"],
                kind=meta["kind"],
                content=results["documents"][0][i],
                context=meta["context"],
                tags=tags,
                impact=meta["impact"],
                alternatives=alternatives,
                embedding=(
                    results["embeddings"][0][i] if results.get("embeddings") else None
                ),
                status=meta["status"],
                source=meta.get("source", ""),
            )
            items.append(item)

        return items

    def touch(self, ids: list[str]) -> None:
        """Bump last_used and uses counts for the given ids."""
        if not ids:
            return

        now = time.time()

        try:
            results = self.collection.get(ids=ids, include=["metadatas"])
            if not results or not results["metadatas"]:
                return

            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                meta["last_used"] = now
                meta["uses"] = meta.get("uses", 0) + 1

                # Chroma requires a full update of metadata
                self.collection.update(ids=[doc_id], metadatas=[meta])
        except Exception as e:
            logger.error(f"Failed to touch memory items: {e}")

    def decay(self, max_idle_seconds: float) -> int:
        """Move items older than max_idle_seconds to dormant state."""
        now = time.time()
        cutoff = now - max_idle_seconds
        decayed_count = 0

        # Check active ids that are loaded in memory
        ids_to_decay = []

        if not self._active_ids:
            return 0

        try:
            results = self.collection.get(
                ids=list(self._active_ids), include=["metadatas"]
            )

            if not results or not results["metadatas"]:
                return 0

            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                last_used = meta.get("last_used", meta.get("created", 0))

                if last_used < cutoff:
                    ids_to_decay.append(doc_id)
                    meta["status"] = "dormant"
                    # Update in chroma
                    self.collection.update(ids=[doc_id], metadatas=[meta])

            for doc_id in ids_to_decay:
                self._active_ids.remove(doc_id)
                decayed_count += 1

        except Exception as e:
            logger.error(f"Failed to decay memory items: {e}")

        return decayed_count

    def all(self) -> list[MemoryItem]:
        """Return all memory items."""
        count = self.collection.count()
        if count == 0:
            return []

        try:
            results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )

            items = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    tags = meta["tags"].split(",") if meta["tags"] else []
                    alternatives = (
                        json.loads(meta["alternatives"])
                        if "alternatives" in meta
                        else []
                    )

                    item = MemoryItem(
                        id=doc_id,
                        created=meta["created"],
                        last_used=meta["last_used"],
                        uses=meta["uses"],
                        kind=meta["kind"],
                        content=results["documents"][i],
                        context=meta["context"],
                        tags=tags,
                        impact=meta["impact"],
                        alternatives=alternatives,
                        embedding=(
                            results["embeddings"][i]
                            if results.get("embeddings")
                            else None
                        ),
                        status=meta["status"],
                        source=meta.get("source", ""),
                    )
                    items.append(item)
            return items

        except Exception as e:
            logger.error(f"Failed to fetch all memory items: {e}")
            return []


# A local memory store matching the spec but using simple JSONL
# Useful for small offline tests where we don't want the Chroma dependency overhead
class LocalMemoryStore:
    def __init__(self, embed_backend, path: str = "results/memory.jsonl"):
        self.embed_backend = embed_backend
        self.path = path
        self.items: dict[str, MemoryItem] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    # Convert list back
                    self.items[data["id"]] = MemoryItem(**data)
        except Exception as e:
            logger.error(f"Failed to load local memory from {self.path}: {e}")

    def _save(self):
        os.makedirs(
            os.path.dirname(self.path) if os.path.dirname(self.path) else ".",
            exist_ok=True,
        )
        try:
            with open(self.path, "w") as f:
                for item in self.items.values():
                    import dataclasses

                    f.write(json.dumps(dataclasses.asdict(item)) + "\n")
        except Exception as e:
            logger.error(f"Failed to save local memory to {self.path}: {e}")

    def write(self, item: MemoryItem) -> None:
        if not item.embedding:
            item.embedding = self.embed_backend.embed([item.content])[0]
        self.items[item.id] = item
        self._save()

    def retrieve(
        self, query: str, k: int, include_dormant: bool = False
    ) -> list[MemoryItem]:
        if not self.items:
            return []

        query_emb = self.embed_backend.embed([query])[0]

        import numpy as np

        candidates = []
        for item in self.items.values():
            if not include_dormant and item.status == "dormant":
                continue

            if not item.embedding:
                continue

            # Cosine similarity
            q = np.array(query_emb)
            i = np.array(item.embedding)
            sim = np.dot(q, i) / (np.linalg.norm(q) * np.linalg.norm(i))
            candidates.append((sim, item))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in candidates[:k]]

    def touch(self, ids: list[str]) -> None:
        now = time.time()
        for doc_id in ids:
            if doc_id in self.items:
                self.items[doc_id].last_used = now
                self.items[doc_id].uses += 1
        self._save()

    def decay(self, max_idle_seconds: float) -> int:
        now = time.time()
        cutoff = now - max_idle_seconds
        count = 0

        for item in self.items.values():
            if item.status == "active" and item.last_used < cutoff:
                item.status = "dormant"
                count += 1

        if count > 0:
            self._save()

        return count

    def all(self) -> list[MemoryItem]:
        return list(self.items.values())


def build_memory(embed_backend) -> MemoryStore | None:
    """Build the memory backend based on configuration."""
    backend_type = os.getenv("CS_MEMORY_BACKEND", "none").lower()

    if backend_type == "none":
        return None
    elif backend_type == "chroma":
        path = os.getenv("CS_MEMORY_PATH", "results/memory_db")
        return ChromaMemoryStore(embed_backend, path)
    elif backend_type == "local":
        path = os.getenv("CS_MEMORY_PATH", "results/memory.jsonl")
        return LocalMemoryStore(embed_backend, path)
    else:
        logger.warning(f"Unknown memory backend: {backend_type}, falling back to none")
        return None
