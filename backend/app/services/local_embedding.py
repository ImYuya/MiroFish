"""
Ollama Embeddings based semantic search.

Uses the Ollama OpenAI-compatible /v1/embeddings endpoint for vectorization,
and numpy cosine similarity for search.
"""

import numpy as np
from typing import List, Tuple, Optional

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_embedding')


class LocalEmbedding:
    """Semantic search using Ollama Embeddings."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or Config.EMBEDDING_MODEL_NAME
        base_url = base_url or Config.LLM_BASE_URL
        api_key = api_key or Config.LLM_API_KEY or "ollama"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LocalEmbedding initialized: model={self.model}, base_url={base_url}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using Ollama /v1/embeddings."""
        if not texts:
            return []

        # Process in batches to avoid overly large requests
        batch_size = 64
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)
            except Exception as e:
                logger.error(f"Embedding failed for batch {i // batch_size}: {e}")
                # Return zero vectors as fallback
                dim = len(all_embeddings[0]) if all_embeddings else 768
                for _ in batch:
                    all_embeddings.append([0.0] * dim)

        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        results = self.embed([text])
        return results[0] if results else []

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def search(
        self,
        query: str,
        documents: List[str],
        doc_embeddings: Optional[List[List[float]]] = None,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search documents by cosine similarity.

        Args:
            query: The search query.
            documents: List of document texts.
            doc_embeddings: Pre-computed embeddings (optional, computed if not provided).
            top_k: Number of top results to return.

        Returns:
            List of (document_index, similarity_score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        query_embedding = self.embed_single(query)
        if not query_embedding or all(v == 0.0 for v in query_embedding):
            return []

        if doc_embeddings is None:
            doc_embeddings = self.embed(documents)

        # Compute similarities
        scores: List[Tuple[int, float]] = []
        for i, doc_emb in enumerate(doc_embeddings):
            if doc_emb and not all(v == 0.0 for v in doc_emb):
                sim = self.cosine_similarity(query_embedding, doc_emb)
                scores.append((i, sim))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
