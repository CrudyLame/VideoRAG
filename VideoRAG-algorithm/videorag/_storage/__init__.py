"""Storage backends for the cloud-native VideoRAG pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from supabase import Client, create_client


@dataclass(slots=True)
class SupabaseVectorStore:
    """Thin async wrapper around Supabase vector storage."""

    url: str
    key: str
    table_name: str
    match_rpc: str
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = create_client(self.url, self.key)

    async def upsert_records(self, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        await asyncio.to_thread(self._upsert_sync, list(records))

    def _upsert_sync(self, records: list[dict[str, Any]]) -> None:
        self.client.table(self.table_name).upsert(records).execute()

    async def similarity_search(
        self,
        session_id: str,
        embedding: Sequence[float],
        match_count: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query_embedding": list(embedding),
            "match_count": match_count,
            "filter": {"session_id": session_id},
        }
        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        response = await asyncio.to_thread(self._rpc_sync, payload)
        return response.data or []

    def _rpc_sync(self, payload: dict[str, Any]):
        return self.client.rpc(self.match_rpc, payload).execute()

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, session_id)

    def _delete_sync(self, session_id: str) -> None:
        self.client.table(self.table_name).delete().eq("session_id", session_id).execute()


__all__ = ["SupabaseVectorStore"]
