"""Prompt templates for PolicyLens RAG operations.

All prompts enforce:
  1. Cite-or-abstain: answer ONLY from provided context.
  2. Verbatim quote extraction for every citation.
  3. JSON-structured output via Pydantic response_model.
"""

from app.core.schemas import PolicyChunk

# ── Shared system instruction ────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are PolicyLens, an expert Indian health insurance policy analyser.

RULES:
1. Answer ONLY using the CONTEXT provided below. Do NOT use external knowledge.
2. For every factual claim, cite the exact clause_id and page number.
3. quote must be a verbatim or near-verbatim excerpt (60 words max) directly from the context.
4. If the answer cannot be found in the context, set is_abstention=true and confidence=0.0.
5. Never invent clause numbers, waiting periods, amounts, or exclusions.
6. Respond ONLY with valid JSON matching the requested schema.
"""

# ── Q&A Prompt ───────────────────────────────────────────────────────────────


def build_qa_prompt(query: str, chunks: list[PolicyChunk]) -> str:
    """Build a grounded Q&A prompt from retrieved chunks."""
    context_parts: list[str] = []
    for c in chunks:
        context_parts.append(
            f"[Clause {c.clause_id} | {c.section} | Page {c.page_start}-{c.page_end}]\n"
            f"{c.text}\n"
        )
    context = "\n---\n".join(context_parts)

    return f"""CONTEXT (policy clauses):
{context}

USER QUESTION: {query}

Instructions:
- Synthesise a clear answer from the above clauses.
- Provide citations (clause_id, page, verbatim quote max 60 words) for each supporting clause.
- confidence: 0.0-1.0 reflecting how well the context covers the question.
- If no clause answers the question, set is_abstention=true.
"""


# ── Policy Summary Prompt ────────────────────────────────────────────────────


def build_summary_prompt(chunks: list[PolicyChunk]) -> str:
    """Build a prompt to extract a structured policy summary card."""
    context_parts: list[str] = []
    for c in chunks:
        context_parts.append(
            f"[Clause {c.clause_id} | {c.section} | Page {c.page_start}]\n{c.text}"
        )
    context = "\n---\n".join(context_parts[:40])  # cap to avoid token overflow

    return f"""CONTEXT (policy clauses):
{context}

Extract a structured policy summary as JSON with these exact fields:
- waiting_period_ped (str): Waiting period for pre-existing diseases (e.g. "36 months")
- waiting_period_specific (str): Initial/specific illness waiting period
- room_rent_cap (str): Room rent sublimit (e.g. "1% of SI per day" or "No cap")
- copay_percent (str): Co-pay percentage if any (e.g. "10%" or "None")
- restoration_benefit (str): Restore/reinstatement clause summary
- no_claim_bonus (str): NCB details
- key_sublimits (list[str]): List of disease/procedure sublimits
- key_exclusions (list[str]): Top 8-10 permanent exclusions
- claim_process_summary (str): Brief cashless/reimbursement claim steps
- policy_name (str): Official policy name from document
- insurer (str): Insurer name

Respond ONLY with valid JSON.
"""


# ── Red Flags Prompt ─────────────────────────────────────────────────────────


def build_red_flags_prompt(chunks: list[PolicyChunk]) -> str:
    """Build a prompt to identify restrictive/unfavourable policy clauses."""
    exclusion_chunks = [
        c for c in chunks if c.chunk_type.value in ("exclusion", "waiting_period", "sub_limit")
    ]
    context_chunks = exclusion_chunks[:25] or chunks[:25]

    context_parts: list[str] = []
    for c in context_chunks:
        context_parts.append(
            f"[Clause {c.clause_id} | {c.section} | Page {c.page_start}]\n{c.text}"
        )
    context = "\n---\n".join(context_parts)

    return f"""CONTEXT (policy exclusion / restriction clauses):
{context}

Identify RED FLAGS that are unusually restrictive, buyer-unfriendly, or commonly misunderstood.
For each red flag return:
- flag_type (str): short category e.g. "High co-pay", "Long waiting period", "Disease sub-limit"
- clause_id (str): source clause
- page (int): page number
- description (str): Plain-language explanation of why this is unfavourable (1-2 sentences)
- severity (str): "high" | "medium" | "low"

Respond as JSON: {{ "red_flags": [ ... ] }}
"""


# ── Comparison Prompt ────────────────────────────────────────────────────────


def build_comparison_prompt(
    scenario: str,
    chunks_a: list[PolicyChunk],
    policy_name_a: str,
    chunks_b: list[PolicyChunk],
    policy_name_b: str,
) -> str:
    """Build a prompt to compare two policies on a given scenario."""

    def fmt(chunks: list[PolicyChunk], name: str) -> str:
        parts = []
        for c in chunks[:15]:
            parts.append(
                f"  [{c.clause_id} | Page {c.page_start}] {c.text[:400]}"
            )
        return f"=== {name} ===\n" + "\n---\n".join(parts)

    ctx_a = fmt(chunks_a, policy_name_a)
    ctx_b = fmt(chunks_b, policy_name_b)

    return f"""SCENARIO: {scenario}

POLICY A:
{ctx_a}

POLICY B:
{ctx_b}

Compare both policies for the above scenario. Return JSON:
{{
  "scenario": "{scenario}",
  "policy_a_name": "{policy_name_a}",
  "policy_b_name": "{policy_name_b}",
  "comparison_points": [
    {{
      "aspect": "<e.g. Coverage, Waiting Period, Sublimit, Exclusion>",
      "policy_a_summary": "<what policy A says, with clause_id>",
      "policy_a_clause_id": "<clause id>",
      "policy_a_page": 0,
      "policy_b_summary": "<what policy B says, with clause_id>",
      "policy_b_clause_id": "<clause id>",
      "policy_b_page": 0,
      "verdict": "A is better or B is better or Similar or Insufficient data"
    }}
  ],
  "overall_verdict": "<1-2 sentence plain-language recommendation>"
}}
Only use information from the provided context.
"""
