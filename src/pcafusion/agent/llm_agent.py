"""LLM-orchestrated variant of the decision-support agent (optional).

Claude decides which tools to call and writes the summary; the Python side
owns every number (all quantities come from tool outputs) and enforces two
guardrails after generation:

* citation grounding: every [ID] cited must have been returned by
  `retrieve_evidence` during this run, otherwise the model is asked to revise;
* audit trail: every tool call and result is kept in `ToolBox.trace`.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import anthropic

from .knowledge import EvidenceRetriever
from .orchestrator import DISCLAIMER, AgentResult, check_citations, triage_flag
from .tools import TOOL_SCHEMAS, ToolBox, dumps

MODEL = "claude-opus-5"
MAX_TURNS = 12

SYSTEM_PROMPT = """You are a research decision-support agent for prostate cancer diagnosis, working on \
the public PI-CAI dataset. For the case you are given, use the tools to gather the patient's available \
modalities, the model's calibrated risk estimate and uncertainty, what drives it, similar cases, and \
relevant guideline evidence. Then write a concise Markdown summary for a clinician with these sections: \
Data used (note any missing modality: it was excluded, not imputed), Risk estimate and uncertainty, \
What drives the estimate, Similar cases, Relevant evidence, Limitations.

Rules:
- Use LOW only if the conformal set is 'no csPCa', no modality is missing, and the probability is below the 95%-sensitivity threshold; otherwise explain why the case is escalated.
- Report only numbers that appear in tool outputs.
- Cite evidence as [ID] using only ids returned by retrieve_evidence in this conversation.
- Do not give a treatment or biopsy recommendation. Give a triage flag instead: LOW, ELEVATED, or \
UNCERTAIN. Use UNCERTAIN whenever the conformal set is 'ambiguous' or 'atypical' or ensemble \
disagreement is high, and say the case should go to multidisciplinary review.
- The system is a research demonstrator; say so briefly."""


class LLMDecisionSupportAgent:
    def __init__(self, toolbox: ToolBox | None = None, model: str = MODEL):
        self.tools = toolbox or ToolBox()
        self.client = anthropic.Anthropic()
        self.model = model

    def _create(self, messages: list):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            thinking={"type": "adaptive"},
            messages=messages,
            # Server-side refusal fallback: if the primary model declines, another model continues.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Model refused: {getattr(response.stop_details, 'category', None)}")
        return response

    def _loop(self, messages: list) -> str:
        for _ in range(MAX_TURNS):
            response = self._create(messages)
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                return "\n".join(b.text for b in response.content if b.type == "text")
            results = []
            for block in tool_uses:
                out = self.tools.call(block.name, **dict(block.input))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": dumps(out),
                                "is_error": "error" in out})
            messages.append({"role": "user", "content": results})
        raise RuntimeError(f"Agent did not finish within {MAX_TURNS} turns")

    def run(self, case_id: str) -> AgentResult:
        self.tools.trace, self.tools.retrieved_ids = [], set()
        known = set(EvidenceRetriever().by_id)
        messages = [{"role": "user", "content": f"Prepare the decision-support summary for case {case_id}."}]
        text = self._loop(messages)
        check = check_citations(text, self.tools.retrieved_ids, known)
        if not check["passed"]:
            bad = check["unknown_ids"] + check["not_retrieved_ids"]
            messages.append({"role": "user", "content":
                             f"These citations were not returned by retrieve_evidence in this run: {bad}. "
                             "Retrieve supporting evidence or remove them, then output the full revised summary."})
            text = self._loop(messages)
            check = check_citations(text, self.tools.retrieved_ids, known)
        risk = next((t["output"] for t in self.tools.trace if t["tool"] == "predict_risk"
                     and not t["input"].get("exclude_modalities")), {})
        case = next((t["output"] for t in self.tools.trace if t["tool"] == "get_case"), {})
        # The flag is always recomputed by code from tool outputs, never taken from the LLM text.
        flag = triage_flag(risk, case.get("missing_modalities"))[0] if "conformal_set_90" in risk else "uncertain"
        report = f"{DISCLAIMER}\n\n{text}"
        return AgentResult(case_id, flag, report, list(self.tools.trace), check["cited"], check)
