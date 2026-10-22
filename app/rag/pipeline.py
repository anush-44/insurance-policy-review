"""PolicyLens RAG Pipeline.

Orchestrates:
  1. Hybrid retrieval (Dense + BM25 → RRF)
  2. LLM generation with citation-or-abstain prompts
  3. Confidence thresholding for abstention
  4. Policy summary extraction
  5. Red flag detection
  6. Side-by-side policy comparison
"""

import json
from typing import Any

from app.core.config import settings
from app.core.llm import LLMClient, get_llm_client
from app.core.logging import logger
from app.core.schemas import ChunkType, Citation, GroundedAnswer, PolicyChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore
from app.rag.prompts import (
    SYSTEM_INSTRUCTION,
    build_comparison_prompt,
    build_qa_prompt,
    build_red_flags_prompt,
    build_summary_prompt,
)
from app.rag.schemas import (
    PolicyComparisonResult,
    PolicySummaryCard,
    RedFlagsReport,
)
from app.retrieval.hybrid import HybridRetriever


class RAGPipeline:
    """End-to-end RAG pipeline for PolicyLens.

    Attributes:
        retriever: HybridRetriever combining dense + BM25 search.
        llm: LLMClient for generation (Gemini, OpenAI, or Mock).
        abstention_threshold: Minimum confidence to emit a grounded answer.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedding_model: EmbeddingModel,
        llm: LLMClient | None = None,
        abstention_threshold: float | None = None,
    ) -> None:
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_model=embedding_model,
        )
        self.llm = llm or get_llm_client()
        self.abstention_threshold = (
            abstention_threshold
            if abstention_threshold is not None
            else settings.abstention_threshold
        )

    @staticmethod
    def _keyword_overlap(query: str, text: str) -> int:
        query_tokens = {
            token
            for token in __import__("re").sub(r"[^a-z0-9]+", " ", query.lower()).split()
            if token
        }
        text_tokens = {
            token
            for token in __import__("re").sub(r"[^a-z0-9]+", " ", text.lower()).split()
            if token
        }
        return len(query_tokens & text_tokens)

    def _fallback_grounded_answer(self, query: str, chunks: list[PolicyChunk]) -> GroundedAnswer:
        best = max(chunks, key=lambda c: self._keyword_overlap(query, c.text), default=None)
        if best is None:
            return GroundedAnswer(
                answer="I couldn't find relevant clauses in this policy to answer your question.",
                citations=[],
                confidence=0.0,
                is_abstention=True,
                caveats=["No matching clauses found in the indexed policy."],
                retrieved_chunk_ids=[],
            )

        overlap = self._keyword_overlap(query, best.text)
        if overlap == 0:
            return GroundedAnswer(
                answer="I couldn't verify this from the policy text available in the index.",
                citations=[],
                confidence=0.0,
                is_abstention=True,
                caveats=["The retrieved clauses do not contain enough evidence for the requested question."],
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
            )

        excerpt = best.text.strip()
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."

        return GroundedAnswer(
            answer=f"The policy wording indicates: {best.text[:220].strip()}",
            citations=[
                Citation(
                    clause_id=best.clause_id,
                    page=best.page_start,
                    quote=excerpt,
                )
            ],
            confidence=0.65,
            is_abstention=False,
            caveats=["Answer generated from the closest matching policy clause."],
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
        )

    # ── Q&A ──────────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        policy_id: str | None = None,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        """Answer a user question, citing clause_id + page for every claim.

        Args:
            query: Natural-language user question.
            policy_id: Restrict retrieval to a specific indexed policy.
            top_k: Number of fused chunks to pass to the LLM.

        Returns:
            GroundedAnswer with citations, confidence, and abstention flag.
        """
        limit = top_k or settings.top_k_rerank
        retrieved: list[tuple[PolicyChunk, float]] = self.retriever.retrieve(
            query=query,
            limit=limit,
            policy_id=policy_id,
        )

        chunks = [c for c, _ in retrieved]

        if not chunks:
            logger.warning(f"No chunks retrieved for query: '{query[:60]}'")
            return GroundedAnswer(
                answer="I couldn't find relevant clauses in this policy to answer your question.",
                citations=[],
                confidence=0.0,
                is_abstention=True,
                caveats=["No matching clauses found in the indexed policy."],
                retrieved_chunk_ids=[],
            )

        prompt = build_qa_prompt(query=query, chunks=chunks)

        try:
            result: GroundedAnswer = self.llm.generate_structured(
                prompt=prompt,
                response_model=GroundedAnswer,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
            )
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            return self._fallback_grounded_answer(query=query, chunks=chunks)

        if result.citations:
            result.retrieved_chunk_ids = [c.chunk_id for c in chunks]
            return result

        if result.confidence < self.abstention_threshold or result.is_abstention:
            result.is_abstention = True
            result.confidence = 0.0
            if not result.caveats:
                result.caveats = ["No evidence in the retrieved clauses supports the answer."]
            result.retrieved_chunk_ids = [c.chunk_id for c in chunks]
            return result

        fallback = self._fallback_grounded_answer(query=query, chunks=chunks)
        result.answer = fallback.answer
        result.citations = fallback.citations
        result.confidence = max(result.confidence, fallback.confidence)
        result.is_abstention = fallback.is_abstention
        result.caveats = (result.caveats or []) + fallback.caveats
        result.retrieved_chunk_ids = [c.chunk_id for c in chunks]

        if result.confidence < self.abstention_threshold and not result.is_abstention:
            result.is_abstention = True
            result.caveats.append(
                f"Confidence ({result.confidence:.2f}) is below threshold "
                f"({self.abstention_threshold}). Please verify with the original policy."
            )

        logger.info(
            f"Q&A: confidence={result.confidence:.2f}, "
            f"abstention={result.is_abstention}, citations={len(result.citations)}"
        )
        return result

    # ── Policy Summary ────────────────────────────────────────────────────────

    def summarise(
        self,
        policy_id: str,
        policy_name: str = "",
        insurer: str = "",
    ) -> PolicySummaryCard:
        """Auto-extract a structured policy summary card.

        Retrieves a broad cross-section of all chunk types then calls the LLM
        to fill in the summary card fields.
        """
        # Pull a wide set of chunks representing all clause types
        all_types = [
            ChunkType.WAITING_PERIOD,
            ChunkType.COVERAGE,
            ChunkType.EXCLUSION,
            ChunkType.SUB_LIMIT,
            ChunkType.CLAIMS,
            ChunkType.DEFINITION,
        ]
        retrieved: list[tuple[PolicyChunk, float]] = self.retriever.retrieve(
            query="waiting period room rent co-pay exclusion restoration claim process",
            limit=40,
            policy_id=policy_id,
            chunk_types=all_types,
        )

        chunks = [c for c, _ in retrieved]
        if not chunks:
            # Fallback: retrieve without type filter
            retrieved = self.retriever.retrieve(
                query="coverage exclusion waiting period",
                limit=40,
                policy_id=policy_id,
            )
            chunks = [c for c, _ in retrieved]

        prompt = build_summary_prompt(chunks=chunks)

        try:
            raw_text = self.llm.generate(
                prompt=prompt,
                system_instruction="You are a structured data extractor. Return only valid JSON.",
                temperature=0.0,
            )
            # Strip markdown code fences if present
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            data: dict[str, Any] = json.loads(raw_text)
            card = PolicySummaryCard(**data)
        except Exception as exc:
            logger.error(f"Summary extraction failed: {exc}")
            card = PolicySummaryCard(
                policy_name=policy_name,
                insurer=insurer,
            )

        # Ensure metadata populated from arguments if LLM left them empty
        if not card.policy_name:
            card.policy_name = policy_name
        if not card.insurer:
            card.insurer = insurer

        return card

    # ── Red Flags ─────────────────────────────────────────────────────────────

    def detect_red_flags(self, policy_id: str) -> RedFlagsReport:
        """Identify buyer-unfriendly clauses in a policy.

        Targets exclusion, waiting period, and sub-limit chunk types.
        """
        retrieved: list[tuple[PolicyChunk, float]] = self.retriever.retrieve(
            query="exclusion waiting period co-pay sublimit disease-wise cap",
            limit=30,
            policy_id=policy_id,
            chunk_types=[ChunkType.EXCLUSION, ChunkType.WAITING_PERIOD, ChunkType.SUB_LIMIT],
        )
        chunks = [c for c, _ in retrieved]

        if not chunks:
            retrieved = self.retriever.retrieve(
                query="excluded not covered waiting period",
                limit=30,
                policy_id=policy_id,
            )
            chunks = [c for c, _ in retrieved]

        prompt = build_red_flags_prompt(chunks=chunks)

        try:
            raw_text = self.llm.generate(
                prompt=prompt,
                system_instruction="You are a consumer protection expert. Return only valid JSON.",
                temperature=0.0,
            )
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            data = json.loads(raw_text)
            report = RedFlagsReport(**data)
        except Exception as exc:
            logger.error(f"Red flags detection failed: {exc}")
            report = RedFlagsReport(red_flags=[])

        logger.info(f"Red flags detected: {len(report.red_flags)} for policy_id={policy_id}")
        return report

    # ── Policy Comparison ─────────────────────────────────────────────────────

    def compare(
        self,
        scenario: str,
        policy_id_a: str,
        policy_name_a: str,
        policy_id_b: str,
        policy_name_b: str,
        top_k: int = 15,
    ) -> PolicyComparisonResult:
        """Compare two indexed policies for a given clinical/financial scenario.

        Args:
            scenario: Human description (e.g. 'knee replacement surgery').
            policy_id_a: ID of first indexed policy.
            policy_name_a: Display name for policy A.
            policy_id_b: ID of second indexed policy.
            policy_name_b: Display name for policy B.
            top_k: Chunks per policy passed to LLM.

        Returns:
            PolicyComparisonResult with comparison_points and overall_verdict.
        """
        chunks_a = [
            c
            for c, _ in self.retriever.retrieve(
                query=scenario, limit=top_k, policy_id=policy_id_a
            )
        ]
        chunks_b = [
            c
            for c, _ in self.retriever.retrieve(
                query=scenario, limit=top_k, policy_id=policy_id_b
            )
        ]

        prompt = build_comparison_prompt(
            scenario=scenario,
            chunks_a=chunks_a,
            policy_name_a=policy_name_a,
            chunks_b=chunks_b,
            policy_name_b=policy_name_b,
        )

        try:
            raw_text = self.llm.generate(
                prompt=prompt,
                system_instruction="You are an insurance expert. Return only valid JSON.",
                temperature=0.0,
            )
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            data = json.loads(raw_text)
            result = PolicyComparisonResult(**data)
        except Exception as exc:
            logger.error(f"Comparison failed: {exc}")
            result = PolicyComparisonResult(
                scenario=scenario,
                policy_a_name=policy_name_a,
                policy_b_name=policy_name_b,
                overall_verdict="Comparison could not be completed due to an internal error.",
            )

        logger.info(
            f"Comparison '{scenario}': {len(result.comparison_points)} aspects compared"
        )
        return result
