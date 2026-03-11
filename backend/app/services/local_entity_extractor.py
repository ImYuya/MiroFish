"""
LLM-based entity and relation extraction.

Replaces Zep Cloud's automatic server-side extraction by using Ollama LLM
with structured prompts to extract entities and relationships from text.
"""

import json
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_entity_extractor')


class LocalEntityExtractor:
    """Extract entities and relations from text using LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def extract_from_text(
        self,
        text: str,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text based on ontology.

        Args:
            text: Input text to extract from.
            ontology: Optional ontology definition with entity_types and edge_types.

        Returns:
            Dict with "entities" and "relationships" lists.
        """
        if not text or not text.strip():
            return {"entities": [], "relationships": []}

        # Build ontology description for the prompt
        ontology_desc = self._build_ontology_description(ontology)

        system_prompt = f"""You are an expert at extracting structured information from text.
Extract all entities and relationships from the given text.

{ontology_desc}

Return a JSON object with exactly this structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "labels": ["EntityType"],
      "summary": "brief description of this entity based on the text",
      "attributes": {{}}
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "name": "relationship_type",
      "fact": "natural language description of the relationship"
    }}
  ]
}}

Rules:
1. Entity names should be canonical (e.g., full names for people).
2. Each entity must have at least one label from the ontology types if provided.
3. Each relationship must have a descriptive "fact" field.
4. Extract ALL entities and relationships mentioned in the text.
5. If the text is in Chinese, keep names in their original language.
6. Return ONLY valid JSON, no additional text."""

        user_prompt = f"Extract entities and relationships from this text:\n\n{text}"

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=8192,
            )

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            # Validate structure
            validated_entities = []
            for e in entities:
                if isinstance(e, dict) and e.get("name"):
                    validated_entities.append({
                        "name": str(e["name"]),
                        "labels": e.get("labels", ["Entity"]),
                        "summary": e.get("summary", ""),
                        "attributes": e.get("attributes", {}),
                    })

            validated_rels = []
            for r in relationships:
                if isinstance(r, dict) and r.get("source") and r.get("target"):
                    validated_rels.append({
                        "source": str(r["source"]),
                        "target": str(r["target"]),
                        "name": r.get("name", "RELATED_TO"),
                        "fact": r.get("fact", ""),
                    })

            logger.info(
                f"Extracted {len(validated_entities)} entities, "
                f"{len(validated_rels)} relationships from text ({len(text)} chars)"
            )

            return {
                "entities": validated_entities,
                "relationships": validated_rels,
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def _build_ontology_description(self, ontology: Optional[Dict[str, Any]]) -> str:
        """Build a textual description of the ontology for the LLM prompt."""
        if not ontology:
            return "Extract any entities and relationships you find."

        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            type_names = [et.get("name", "") for et in entity_types]
            parts.append(f"Entity types to look for: {', '.join(type_names)}")

            for et in entity_types:
                desc = et.get("description", "")
                attrs = et.get("attributes", [])
                if desc or attrs:
                    attr_names = [a.get("name", "") for a in attrs]
                    detail = f"- {et['name']}: {desc}"
                    if attr_names:
                        detail += f" (attributes: {', '.join(attr_names)})"
                    parts.append(detail)

        edge_types = ontology.get("edge_types", [])
        if edge_types:
            type_names = [et.get("name", "") for et in edge_types]
            parts.append(f"\nRelationship types to look for: {', '.join(type_names)}")

            for et in edge_types:
                desc = et.get("description", "")
                if desc:
                    parts.append(f"- {et['name']}: {desc}")

        return "\n".join(parts) if parts else "Extract any entities and relationships you find."
