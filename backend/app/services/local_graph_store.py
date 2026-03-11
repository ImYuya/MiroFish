"""
Local graph store using NetworkX + Ollama Embeddings + JSON persistence.

Replaces all Zep Cloud API calls with a fully local implementation.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx

from ..config import Config
from ..utils.logger import get_logger
from .local_embedding import LocalEmbedding
from .local_entity_extractor import LocalEntityExtractor

logger = get_logger('mirofish.local_graph_store')

# Base directory for graph persistence
_GRAPHS_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/graphs')


class _NodeData:
    """In-memory representation matching Zep node fields."""
    __slots__ = ('uuid_', 'name', 'labels', 'summary', 'attributes', 'created_at', 'embedding')

    def __init__(self, uuid_: str, name: str, labels: List[str], summary: str,
                 attributes: Dict[str, Any], created_at: str, embedding: Optional[List[float]] = None):
        self.uuid_ = uuid_
        self.name = name
        self.labels = labels
        self.summary = summary
        self.attributes = attributes
        self.created_at = created_at
        self.embedding = embedding


class _EdgeData:
    """In-memory representation matching Zep edge fields."""
    __slots__ = ('uuid_', 'name', 'fact', 'source_node_uuid', 'target_node_uuid',
                 'attributes', 'created_at', 'valid_at', 'invalid_at', 'expired_at',
                 'episodes', 'fact_type', 'embedding')

    def __init__(self, uuid_: str, name: str, fact: str,
                 source_node_uuid: str, target_node_uuid: str,
                 attributes: Optional[Dict[str, Any]] = None,
                 created_at: Optional[str] = None,
                 valid_at: Optional[str] = None,
                 invalid_at: Optional[str] = None,
                 expired_at: Optional[str] = None,
                 episodes: Optional[List[str]] = None,
                 fact_type: Optional[str] = None,
                 embedding: Optional[List[float]] = None):
        self.uuid_ = uuid_
        self.name = name
        self.fact = fact
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.attributes = attributes or {}
        self.created_at = created_at
        self.valid_at = valid_at
        self.invalid_at = invalid_at
        self.expired_at = expired_at
        self.episodes = episodes or []
        self.fact_type = fact_type or name
        self.embedding = embedding


class LocalGraphStore:
    """
    NetworkX-based local graph store.

    Provides the same interface surface as Zep Cloud graph API so that existing
    services can switch over with minimal changes.
    """

    def __init__(self):
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._ontologies: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Lazy-initialized helpers
        self._embedding: Optional[LocalEmbedding] = None
        self._extractor: Optional[LocalEntityExtractor] = None

        # Ensure storage directory exists
        os.makedirs(_GRAPHS_DIR, exist_ok=True)

    # ------ lazy helpers ------

    @property
    def embedding(self) -> LocalEmbedding:
        if self._embedding is None:
            self._embedding = LocalEmbedding()
        return self._embedding

    @property
    def extractor(self) -> LocalEntityExtractor:
        if self._extractor is None:
            self._extractor = LocalEntityExtractor()
        return self._extractor

    # ========== Graph management ==========

    def create_graph(self, graph_id: str, name: str = "", description: str = "") -> str:
        """Create a new graph (equivalent to Zep graph.create)."""
        with self._lock:
            g = nx.DiGraph()
            g.graph['name'] = name
            g.graph['description'] = description
            g.graph['created_at'] = datetime.now().isoformat()
            self._graphs[graph_id] = g
        logger.info(f"Graph created: {graph_id} ({name})")
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        """Delete a graph and its persisted data (equivalent to Zep graph.delete)."""
        with self._lock:
            self._graphs.pop(graph_id, None)
            self._ontologies.pop(graph_id, None)

        # Remove persisted files
        graph_dir = os.path.join(_GRAPHS_DIR, graph_id)
        if os.path.isdir(graph_dir):
            import shutil
            shutil.rmtree(graph_dir, ignore_errors=True)

        logger.info(f"Graph deleted: {graph_id}")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        """Store ontology definition for a graph (equivalent to Zep graph.set_ontology)."""
        self._ontologies[graph_id] = ontology
        logger.info(f"Ontology set for graph {graph_id}")

    def _ensure_graph(self, graph_id: str) -> nx.DiGraph:
        """Get graph, loading from disk if needed."""
        if graph_id not in self._graphs:
            self._load_graph(graph_id)
        if graph_id not in self._graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        return self._graphs[graph_id]

    # ========== Data ingestion ==========

    def add_episodes(
        self,
        graph_id: str,
        episodes: List[Dict[str, str]],
    ) -> List[str]:
        """
        Add text episodes, extract entities/relations, and update the graph.

        Equivalent to Zep graph.add_batch(). Each episode is {"data": "...", "type": "text"}.
        Returns a list of episode UUIDs.
        """
        g = self._ensure_graph(graph_id)
        ontology = self._ontologies.get(graph_id)
        episode_uuids: List[str] = []

        for ep in episodes:
            text = ep.get("data", "") if isinstance(ep, dict) else str(ep)
            if not text.strip():
                continue

            ep_uuid = uuid.uuid4().hex
            episode_uuids.append(ep_uuid)

            # Extract entities and relations via LLM
            extraction = self.extractor.extract_from_text(text, ontology)

            self._merge_extraction(g, graph_id, extraction, ep_uuid)

        # Persist after batch
        self._save_graph(graph_id)
        logger.info(f"Added {len(episode_uuids)} episodes to graph {graph_id}")
        return episode_uuids

    def add_text(self, graph_id: str, text: str) -> None:
        """
        Add a single text to the graph (equivalent to Zep graph.add).

        Used for incremental updates during simulation.
        """
        if not text or not text.strip():
            return

        g = self._ensure_graph(graph_id)
        ontology = self._ontologies.get(graph_id)
        ep_uuid = uuid.uuid4().hex

        extraction = self.extractor.extract_from_text(text, ontology)
        self._merge_extraction(g, graph_id, extraction, ep_uuid)

        self._save_graph(graph_id)
        logger.debug(f"Added text to graph {graph_id} ({len(text)} chars)")

    def _merge_extraction(
        self, g: nx.DiGraph, graph_id: str,
        extraction: Dict[str, Any], episode_uuid: str,
    ) -> None:
        """Merge extracted entities and relationships into the graph."""
        now = datetime.now().isoformat()
        entity_name_to_uuid: Dict[str, str] = {}

        # Map existing node names to UUIDs
        for nid, ndata in g.nodes(data=True):
            name = ndata.get('name', '')
            if name:
                entity_name_to_uuid[name.lower()] = nid

        # Upsert entities
        for entity in extraction.get("entities", []):
            name = entity.get("name", "")
            if not name:
                continue

            existing_uuid = entity_name_to_uuid.get(name.lower())
            if existing_uuid:
                # Update existing node
                node = g.nodes[existing_uuid]
                # Merge labels
                old_labels = set(node.get('labels', []))
                new_labels = set(entity.get('labels', []))
                node['labels'] = list(old_labels | new_labels)
                # Update summary if new one is longer
                new_summary = entity.get('summary', '')
                if len(new_summary) > len(node.get('summary', '')):
                    node['summary'] = new_summary
                # Merge attributes
                old_attrs = node.get('attributes', {})
                old_attrs.update(entity.get('attributes', {}))
                node['attributes'] = old_attrs
                node_uuid = existing_uuid
            else:
                node_uuid = uuid.uuid4().hex
                labels = entity.get('labels', ['Entity'])
                if 'Entity' not in labels:
                    labels.append('Entity')

                g.add_node(node_uuid,
                           name=name,
                           labels=labels,
                           summary=entity.get('summary', ''),
                           attributes=entity.get('attributes', {}),
                           created_at=now,
                           embedding=None)
                entity_name_to_uuid[name.lower()] = node_uuid

        # Add relationships as edges
        for rel in extraction.get("relationships", []):
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            if not source_name or not target_name:
                continue

            source_uuid = entity_name_to_uuid.get(source_name.lower())
            target_uuid = entity_name_to_uuid.get(target_name.lower())

            # Create nodes for unknown entities
            if not source_uuid:
                source_uuid = uuid.uuid4().hex
                g.add_node(source_uuid, name=source_name, labels=['Entity'],
                           summary='', attributes={}, created_at=now, embedding=None)
                entity_name_to_uuid[source_name.lower()] = source_uuid

            if not target_uuid:
                target_uuid = uuid.uuid4().hex
                g.add_node(target_uuid, name=target_name, labels=['Entity'],
                           summary='', attributes={}, created_at=now, embedding=None)
                entity_name_to_uuid[target_name.lower()] = target_uuid

            edge_uuid = uuid.uuid4().hex
            g.add_edge(
                source_uuid, target_uuid,
                key=edge_uuid,
                uuid_=edge_uuid,
                name=rel.get("name", "RELATED_TO"),
                fact=rel.get("fact", ""),
                fact_type=rel.get("name", "RELATED_TO"),
                attributes={},
                created_at=now,
                valid_at=now,
                invalid_at=None,
                expired_at=None,
                episodes=[episode_uuid],
                embedding=None,
            )

    # ========== Node / Edge retrieval ==========

    def get_nodes(
        self, graph_id: str, limit: int = 2000, cursor: Optional[str] = None,
    ) -> List[_NodeData]:
        """Get nodes from a graph (equivalent to Zep graph.node.get_by_graph_id)."""
        g = self._ensure_graph(graph_id)
        nodes_list = []
        past_cursor = cursor is None

        for nid, ndata in g.nodes(data=True):
            if not past_cursor:
                if nid == cursor:
                    past_cursor = True
                continue

            nodes_list.append(_NodeData(
                uuid_=nid,
                name=ndata.get('name', ''),
                labels=ndata.get('labels', []),
                summary=ndata.get('summary', ''),
                attributes=ndata.get('attributes', {}),
                created_at=ndata.get('created_at', ''),
                embedding=ndata.get('embedding'),
            ))
            if len(nodes_list) >= limit:
                break

        return nodes_list

    def get_node(self, node_uuid: str, graph_id: Optional[str] = None) -> Optional[_NodeData]:
        """Get a single node by UUID (equivalent to Zep graph.node.get)."""
        # Search across all graphs if graph_id is not provided
        graphs_to_search = [self._graphs[graph_id]] if graph_id and graph_id in self._graphs else self._graphs.values()

        for g in graphs_to_search:
            if node_uuid in g.nodes:
                ndata = g.nodes[node_uuid]
                return _NodeData(
                    uuid_=node_uuid,
                    name=ndata.get('name', ''),
                    labels=ndata.get('labels', []),
                    summary=ndata.get('summary', ''),
                    attributes=ndata.get('attributes', {}),
                    created_at=ndata.get('created_at', ''),
                    embedding=ndata.get('embedding'),
                )
        return None

    def get_node_edges(self, node_uuid: str, graph_id: Optional[str] = None) -> List[_EdgeData]:
        """Get all edges connected to a node (equivalent to Zep graph.node.get_entity_edges)."""
        graphs_to_search = (
            [(graph_id, self._graphs[graph_id])] if graph_id and graph_id in self._graphs
            else list(self._graphs.items())
        )

        edges: List[_EdgeData] = []
        for gid, g in graphs_to_search:
            if not isinstance(g, nx.MultiDiGraph) and not isinstance(g, nx.DiGraph):
                continue
            if node_uuid not in g.nodes:
                continue

            # Outgoing edges
            for _, target, edata in g.out_edges(node_uuid, data=True):
                edges.append(self._edge_data_from_dict(edata, node_uuid, target))

            # Incoming edges
            for source, _, edata in g.in_edges(node_uuid, data=True):
                edges.append(self._edge_data_from_dict(edata, source, node_uuid))

        return edges

    def get_edges(
        self, graph_id: str, limit: int = 10000, cursor: Optional[str] = None,
    ) -> List[_EdgeData]:
        """Get edges from a graph (equivalent to Zep graph.edge.get_by_graph_id)."""
        g = self._ensure_graph(graph_id)
        edges_list: List[_EdgeData] = []
        past_cursor = cursor is None

        for source, target, edata in g.edges(data=True):
            edge_uuid = edata.get('uuid_', '')
            if not past_cursor:
                if edge_uuid == cursor:
                    past_cursor = True
                continue

            edges_list.append(self._edge_data_from_dict(edata, source, target))
            if len(edges_list) >= limit:
                break

        return edges_list

    @staticmethod
    def _edge_data_from_dict(edata: Dict[str, Any], source: str, target: str) -> _EdgeData:
        return _EdgeData(
            uuid_=edata.get('uuid_', ''),
            name=edata.get('name', ''),
            fact=edata.get('fact', ''),
            source_node_uuid=source,
            target_node_uuid=target,
            attributes=edata.get('attributes', {}),
            created_at=edata.get('created_at'),
            valid_at=edata.get('valid_at'),
            invalid_at=edata.get('invalid_at'),
            expired_at=edata.get('expired_at'),
            episodes=edata.get('episodes', []),
            fact_type=edata.get('fact_type', ''),
            embedding=edata.get('embedding'),
        )

    # ========== Semantic search ==========

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "rrf",
    ) -> Dict[str, Any]:
        """
        Semantic search over the graph (equivalent to Zep graph.search).

        Returns an object with .edges and .nodes attributes (SimpleNamespace)
        to match Zep SDK response format.
        """
        from types import SimpleNamespace

        g = self._ensure_graph(graph_id)
        result_edges: List[Any] = []
        result_nodes: List[Any] = []

        if scope in ("edges", "both"):
            result_edges = self._search_edges(g, graph_id, query, limit)

        if scope in ("nodes", "both"):
            result_nodes = self._search_nodes(g, graph_id, query, limit)

        return SimpleNamespace(edges=result_edges, nodes=result_nodes)

    def _search_edges(self, g: nx.DiGraph, graph_id: str, query: str, limit: int) -> list:
        """Semantic + keyword search over edges."""
        from types import SimpleNamespace

        all_edges_data = []
        for source, target, edata in g.edges(data=True):
            fact = edata.get('fact', '')
            if fact:
                all_edges_data.append((source, target, edata, fact))

        if not all_edges_data:
            return []

        # Try semantic search first
        facts = [item[3] for item in all_edges_data]
        ranked = self._semantic_and_keyword_search(facts, query, limit)

        results = []
        for idx, score in ranked:
            source, target, edata, fact = all_edges_data[idx]
            results.append(SimpleNamespace(
                uuid_=edata.get('uuid_', ''),
                uuid=edata.get('uuid_', ''),
                name=edata.get('name', ''),
                fact=fact,
                fact_type=edata.get('fact_type', ''),
                source_node_uuid=source,
                target_node_uuid=target,
                attributes=edata.get('attributes', {}),
                created_at=edata.get('created_at'),
                valid_at=edata.get('valid_at'),
                invalid_at=edata.get('invalid_at'),
                expired_at=edata.get('expired_at'),
                episodes=edata.get('episodes', []),
                score=score,
                content=fact,
            ))

        return results

    def _search_nodes(self, g: nx.DiGraph, graph_id: str, query: str, limit: int) -> list:
        """Semantic + keyword search over nodes."""
        from types import SimpleNamespace

        all_nodes_data = []
        for nid, ndata in g.nodes(data=True):
            text = f"{ndata.get('name', '')} {ndata.get('summary', '')}"
            if text.strip():
                all_nodes_data.append((nid, ndata, text))

        if not all_nodes_data:
            return []

        texts = [item[2] for item in all_nodes_data]
        ranked = self._semantic_and_keyword_search(texts, query, limit)

        results = []
        for idx, score in ranked:
            nid, ndata, text = all_nodes_data[idx]
            results.append(SimpleNamespace(
                uuid_=nid,
                uuid=nid,
                name=ndata.get('name', ''),
                labels=ndata.get('labels', []),
                summary=ndata.get('summary', ''),
                attributes=ndata.get('attributes', {}),
                created_at=ndata.get('created_at', ''),
                score=score,
                content=ndata.get('summary', ''),
            ))

        return results

    def _semantic_and_keyword_search(
        self, documents: List[str], query: str, limit: int,
    ) -> List[Tuple[int, float]]:
        """
        Combined semantic + keyword search with RRF-like fusion.

        Falls back to keyword-only if embedding fails.
        """
        # Keyword scoring
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        keyword_scores: List[Tuple[int, float]] = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            score = 0.0
            if query_lower in doc_lower:
                score += 100.0
            for kw in keywords:
                if kw in doc_lower:
                    score += 10.0
            keyword_scores.append((i, score))

        # Try semantic search
        semantic_scores: List[Tuple[int, float]] = []
        try:
            semantic_scores = self.embedding.search(query, documents, top_k=min(limit * 2, len(documents)))
        except Exception as e:
            logger.warning(f"Semantic search failed, using keyword-only: {e}")

        # RRF fusion
        if semantic_scores:
            # Build rank maps
            keyword_rank = {}
            sorted_kw = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
            for rank, (idx, _) in enumerate(sorted_kw):
                keyword_rank[idx] = rank + 1

            semantic_rank = {}
            for rank, (idx, _) in enumerate(semantic_scores):
                semantic_rank[idx] = rank + 1

            k = 60  # RRF constant
            all_indices = set(idx for idx, _ in keyword_scores) | set(idx for idx, _ in semantic_scores)
            fused: List[Tuple[int, float]] = []
            n = len(documents)
            for idx in all_indices:
                kr = keyword_rank.get(idx, n + 1)
                sr = semantic_rank.get(idx, n + 1)
                rrf_score = 1.0 / (k + kr) + 1.0 / (k + sr)
                fused.append((idx, rrf_score))

            fused.sort(key=lambda x: x[1], reverse=True)
            return fused[:limit]
        else:
            # Keyword-only fallback
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            return [(idx, score) for idx, score in keyword_scores if score > 0][:limit]

    # ========== Persistence ==========

    def _graph_dir(self, graph_id: str) -> str:
        return os.path.join(_GRAPHS_DIR, graph_id)

    def _save_graph(self, graph_id: str) -> None:
        """Persist graph to JSON."""
        if graph_id not in self._graphs:
            return

        g = self._graphs[graph_id]
        graph_dir = self._graph_dir(graph_id)
        os.makedirs(graph_dir, exist_ok=True)

        data = {
            "graph_meta": dict(g.graph),
            "nodes": [],
            "edges": [],
            "ontology": self._ontologies.get(graph_id),
        }

        for nid, ndata in g.nodes(data=True):
            node_dict = dict(ndata)
            node_dict['uuid_'] = nid
            # Don't persist embeddings in the main JSON to keep it small
            node_dict.pop('embedding', None)
            data["nodes"].append(node_dict)

        for source, target, edata in g.edges(data=True):
            edge_dict = dict(edata)
            edge_dict['source_node_uuid'] = source
            edge_dict['target_node_uuid'] = target
            edge_dict.pop('embedding', None)
            data["edges"].append(edge_dict)

        json_path = os.path.join(graph_dir, 'graph.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Graph saved: {graph_id} ({len(data['nodes'])} nodes, {len(data['edges'])} edges)")

    def _load_graph(self, graph_id: str) -> None:
        """Load graph from JSON."""
        json_path = os.path.join(self._graph_dir(graph_id), 'graph.json')
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            g = nx.DiGraph()
            g.graph.update(data.get("graph_meta", {}))

            for node_dict in data.get("nodes", []):
                nid = node_dict.pop('uuid_')
                g.add_node(nid, **node_dict)

            for edge_dict in data.get("edges", []):
                source = edge_dict.pop('source_node_uuid')
                target = edge_dict.pop('target_node_uuid')
                g.add_edge(source, target, **edge_dict)

            with self._lock:
                self._graphs[graph_id] = g
                if data.get("ontology"):
                    self._ontologies[graph_id] = data["ontology"]

            logger.info(f"Graph loaded from disk: {graph_id} "
                        f"({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")

        except Exception as e:
            logger.error(f"Failed to load graph {graph_id}: {e}")


# Singleton instance for application-wide use
_store_instance: Optional[LocalGraphStore] = None
_store_lock = threading.Lock()


def get_graph_store() -> LocalGraphStore:
    """Get the singleton LocalGraphStore instance."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = LocalGraphStore()
    return _store_instance
